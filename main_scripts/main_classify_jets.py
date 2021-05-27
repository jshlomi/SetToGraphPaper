import random
import os
import sys
import argparse
import copy
import shutil
import json
from pprint import pprint
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


"""
How To:
Example for running from command line:
python <path_to>/SetToGraph/main_scripts/main_jets.py
"""
# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from models.set_to_graph import SetToGraph
from models.set_to_graph_siam import SetToGraphSiam
from dataloaders import jets_loader
#from performance_eval.eval_test_jets import eval_jets_on_test_set
from models.classifier import JetClassifier


DEVICE = 'cuda'


def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-g', '--gpu', default='0', help='The gpu to run on')
    argparser.add_argument('-e', '--epochs', default=400, type=int, help='The number of epochs to run')
    argparser.add_argument('-l', '--lr', default=0.0005, type=float, help='The learning rate')
    argparser.add_argument('-b', '--bs', default=1000, type=int, help='Batch size to use')
                                                #2048
    argparser.add_argument('--res_dir', default='../experiments/jets_results', help='Results directory')
    argparser.add_argument('--pretrained_vertexing_model', default=None, help='path to trained model')
    argparser.add_argument('--pretrained_vertexing_model_type', default=None, help='s2g, rnn or siam')

    argparser.add_argument('--use_rave', dest='use_rave', action='store_true')

    argparser.add_argument('--debug_load', dest='debug_load', action='store_true', help='Load only a small subset of the data')
    argparser.add_argument('--save', dest='save', action='store_true', help='Whether to save all to disk')
    argparser.add_argument('--no_save', dest='save', action='store_false')
    argparser.set_defaults(save=True, debug_load=False,use_rave=False)

    args = argparser.parse_args()



    return args


def calc_metrics(jet_prediction, jet_label, accum_info,batch_size):
    with torch.no_grad():
        
        pred = torch.argmax(jet_prediction,dim=1)

        for flav,flav_name in zip([0,1,2],['b','c','u']):
            correct = len(torch.where(pred[jet_label==flav]==jet_label[jet_label==flav])[0])

            total = len(jet_label[jet_label==flav])

            accum_info['accuracy_'+flav_name]+= correct/total
        
        accum_info['accuracy'] +=  len(torch.where(pred==jet_label)[0])
        

    return accum_info


def get_loss(y_hat, y):
    
    loss = F.cross_entropy(y_hat,y)

    return loss


def train(data, model, optimizer,use_rave=False):
    train_info = do_epoch(data, model, optimizer,use_rave)
    return train_info


def evaluate(data, model,use_rave=False):
    val_info = do_epoch(data, model, optimizer=None,use_rave=use_rave)
    return val_info


def do_epoch(data, model, optimizer=None,use_rave=False):
    if optimizer is not None:
        # train epoch
        model.train()
    else:
        # validation epoch
        model.eval()
    start_time = datetime.now()

    # Iterate over batches
    accum_info = {k: 0.0 for k in ['loss','accuracy','insts','accuracy_b','accuracy_c','accuracy_u']}
    n_batches = 0
    for batch in data:

        if use_rave:
            sets, _, _, jet_features, jet_label, rave_input = batch
        else:
            sets, _, _, jet_features, jet_label = batch

  
        # One Train step on the current batch
        sets = sets.to(DEVICE, torch.float)
        jet_label = jet_label.to(DEVICE,torch.long)
        batch_size = sets.shape[0]
        accum_info['insts'] += batch_size
        n_batches+=1
        
        if use_rave:
            jet_prediction = model(jet_features,sets,external_edge_vals=rave_input)  # B,N,N
        else:
            jet_prediction = model(jet_features,sets)
        loss = get_loss(jet_prediction, jet_label)

        if optimizer is not None:
            # backprop for training epochs only
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calc ri
        accum_info = calc_metrics(jet_prediction, jet_label, accum_info,batch_size)

        # update results from train_step func
        accum_info['loss'] += loss.item() * batch_size

    num_insts = accum_info.pop('insts')
    accum_info['loss'] /= num_insts
    accum_info['accuracy'] /= num_insts
    for flav,flav_name in zip([0,1,2],['b','c','u']):
        
        accum_info['accuracy_'+flav_name] /= n_batches

    accum_info['run_time'] = datetime.now() - start_time
    accum_info['run_time'] = str(accum_info['run_time']).split(".")[0]

    return accum_info


def main():
    start_time = datetime.now()

    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # can impact performance
    # torch.backends.cudnn.benchmark = False  # can impact performance

    config = parse_args()

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # uncomment only for CUDA error debugging
    os.environ["CUDA_VISIBLE_DEVICES"] = '1' #config.gpu
    #torch.cuda.set_device(int(config.gpu))

    pprint(vars(config))
    print(flush=True)

    # Load data
    print('Loading training data...', end='', flush=True)
    train_data = jets_loader.get_data_loader('train', config.bs, config.debug_load,add_jet_flav=True,add_rave_file=config.use_rave)
    print('Loading validation data...', end='', flush=True)
    val_data = jets_loader.get_data_loader('validation', config.bs, config.debug_load,add_jet_flav=True,add_rave_file=config.use_rave)
                                            #validation
    # Create model instance
    if config.pretrained_vertexing_model_type == 'rnn':
        vertexing_config = {
            'in_features' : 10,
            'out_features' :1,
            'set_fn_feats' : [256, 256,128, 6],
            'method' : 'lin5',
            'hidden_mlp' : [256],
            'predict_diagonal' : False,
            'attention' : True,
            'set_model_type' : 'RNN'
            
        }
    elif config.pretrained_vertexing_model_type == 'siam':
        vertexing_config = {
            'in_features' : 10,
            'set_fn_feats' : [384, 384, 384, 384, 5],
            'hidden_mlp' : [256],
        }
        
    elif config.pretrained_vertexing_model_type == 's2g':
        vertexing_config = {
            'in_features' : 10,
            'out_features' :1,
            'set_fn_feats' : [256, 256, 256, 256, 5],
            'method' : 'lin5',
            'hidden_mlp' : [256],
            'predict_diagonal' : False,
            'attention' : True,
            'set_model_type' : 'deepset'
            
        }
    else:
        vertexing_config = {}
    
    model = JetClassifier(10,vertexing_config,vertexing_type=config.pretrained_vertexing_model_type);
    if config.pretrained_vertexing_model!=None:
        vertexing_model_state_dict = torch.load(config.pretrained_vertexing_model,map_location='cpu')
        state_dict = model.state_dict()
        for key in vertexing_model_state_dict.keys():
            state_dict['vertexing.'+key] = vertexing_model_state_dict[key]

        model.load_state_dict(state_dict)
        for name,p in model.named_parameters():
            if 'vertexing.' in name:
                p.requires_grad = False




    
    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The nubmer of model parameters is {num_params}')

    # Optimizer

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)

    # Metrics
    train_loss = np.empty(config.epochs, float)
    train_acc = np.empty(config.epochs, float)
    val_loss = np.empty(config.epochs, float)
    val_acc = np.empty(config.epochs, float)

    best_epoch = -1
    best_val_acc = -1
    best_model = None

    # Training and evaluation process
    for epoch in range(1, config.epochs + 1):
        train_info = train(train_data, model, optimizer,use_rave=config.use_rave)
        print(f"\tTraining - {epoch:4}",
              " loss:{loss:.6f} -- mean_acc:{accuracy:.4f} -- mean_b_acc:{accuracy_b:.4f} -- mean_c_acc:{accuracy_c:.4f} -- mean_light_acc:{accuracy_u:.4f}".format(**train_info), flush=True)
        train_loss[epoch-1], train_acc[epoch-1] = train_info['loss'], train_info['accuracy']

        val_info = evaluate(val_data, model,use_rave=config.use_rave)
        print(f"\tVal      - {epoch:4}",
              " loss:{loss:.6f} -- mean_acc:{accuracy:.4f} -- mean_b_acc:{accuracy_b:.4f} -- mean_c_acc:{accuracy_c:.4f} -- mean_light_acc:{accuracy_u:.4f} -- runtime:{run_time}\n".format(**val_info), flush=True)
        val_loss[epoch-1], val_acc[epoch-1] = val_info['loss'], val_info['accuracy']

        if val_info['accuracy'] > best_val_acc:
            best_val_acc = val_info['accuracy']
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        
        if best_epoch < epoch - 20:
            print('Early stopping training due to no improvement over the last 20 epochs...')
            break

    del train_data, val_data
    print(f'Best validation acc: {best_val_acc:.4f}, best epoch: {best_epoch}.')

    print(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')
    print()

    # Saving to disk
    if config.save:
        if not os.path.isdir(config.res_dir):
            os.makedirs(config.res_dir)
        exp_dir = f'jets_{start_time:%Y%m%d_%H%M%S}_0'
        output_dir = os.path.join(config.res_dir, exp_dir)

        i = 0
        while True:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)  # raises error if dir already exists
                break
            i += 1
            output_dir = output_dir[:-1] + str(i)
            if i > 9:
                print(f'Cannot save results on disk. (tried to save as {output_dir})')
                return

        print(f'Saving all to {output_dir}')
        torch.save(best_model.state_dict(), os.path.join(output_dir, "exp_model.pt"))
        shutil.copyfile(__file__, os.path.join(output_dir, 'code.py'))
        results_dict = {'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc}
        df = pd.DataFrame(results_dict)
        df.index.name = 'epochs'
        df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        best_dict = {'best_val_acc': best_val_acc, 'best_epoch': best_epoch}
        best_df = pd.DataFrame(best_dict, index=[0])
        best_df.to_csv(os.path.join(output_dir, "best_val_results.csv"), index=False)
        with open(os.path.join(output_dir, 'used_config.json'), 'w') as fp:
            json.dump(vars(config), fp)

    # print('Loading test data...', end='', flush=True)
    # test_data = jets_loader.get_data_loader('test', config.bs, config.debug_load)
    # test_info = evaluate(test_data, best_model)
    # print(f"\tTest     - {best_epoch:4}",
    #       " loss:{loss:.6f} -- mean_ri:{ri:.4f} -- fscore:{fscore:.4f} -- recall:{recall:.4f} "
    #       "-- precision:{precision:.4f}  -- runtime:{run_time}\n".format(**test_info))

    print(f'Epoch {best_epoch} - evaluating over test set.')
    #test_results = eval_jets_on_test_set(best_model)
    #print('Test results:')
    # print(test_results)
    # if config.save:
    #     test_results.to_csv(os.path.join(output_dir, "test_results.csv"), index=True)

    print(f'Total runtime: {str(datetime.now() - start_time).split(".")[0]}')


if __name__ == '__main__':
    main()
