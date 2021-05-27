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
from tqdm import tqdm

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
	
	argparser.add_argument('-b', '--bs', default=1000, type=int, help='Batch size to use')
												#2048

	argparser.add_argument('--vertexing_model_type')
	argparser.add_argument('--path_to_trained_model', default=None, help='path to trained model')
	argparser.add_argument('--outputfilename')

	argparser.add_argument('--use_rave', dest='use_rave', action='store_true')
	argparser.set_defaults(use_rave=False)

	args = argparser.parse_args()



	return args









def evaluate(data, model,use_rave=False):

	model.eval()


	all_jet_predictions = []

	for batch in data:

		if use_rave:
			sets, _, _, jet_features, jet_label, rave_input = batch
			sets = sets.to(DEVICE, torch.float)
			jet_prediction = model(jet_features,sets,rave_input).cpu().data.numpy() # B,N,N
		else:
			sets, _, _, jet_features, jet_label = batch
			sets = sets.to(DEVICE, torch.float)
			jet_prediction = model(jet_features,sets).cpu().data.numpy() # B,N,N
		
		
		
	  
		
		all_jet_predictions.append(jet_prediction)
   

	return np.concatenate(all_jet_predictions)


def main():
	start_time = datetime.now()

   

	config = parse_args()

	# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # uncomment only for CUDA error debugging
	os.environ["CUDA_VISIBLE_DEVICES"] = '1' #config.gpu
	#torch.cuda.set_device(int(config.gpu))



	# Load data
	print('Loading test data...', end='', flush=True)
	test_data = jets_loader.JetGraphDataset('test',debug_load=False,add_jet_flav=True,add_rave_file=config.use_rave)
   
											#validation
	# Create model instance
	if config.vertexing_model_type == 'rnn':
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
	elif config.vertexing_model_type == 'siam':
		vertexing_config = {
			'in_features' : 10,
			'set_fn_feats' : [384, 384, 384, 384, 5],
			'hidden_mlp' : [256],
		}
		
	elif config.vertexing_model_type == 's2g':
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
	
	model = JetClassifier(10,vertexing_config,vertexing_type=config.vertexing_model_type)
	model.load_state_dict( torch.load(config.path_to_trained_model,map_location='cpu') )
	model.eval()




	
	model = model.to(DEVICE)
   
	predictions = []
	indx_list = []

	max_batch_size = 1000

	for tracks_in_jet in tqdm(range(2, np.amax(test_data.n_nodes)+1)):
		trk_indxs = np.where(np.array(test_data.n_nodes) == tracks_in_jet)[0]
		if len(trk_indxs) < 1:
			continue
		indx_list += list(trk_indxs)


		n_sub_batches = len(trk_indxs)//max_batch_size+1

		sub_batches = np.array_split(trk_indxs,n_sub_batches)
		for sub_batch in sub_batches:
			if config.use_rave:
				sets = []
				jet_features = []
				rave_inputs = []

				for i in sub_batch:
					sets_i, _, _, jet_features_i, _, rave_input = test_data[i]
					sets.append(torch.tensor(sets_i))
					jet_features.append(torch.tensor(jet_features_i))
					rave_inputs.append(rave_input)

				sets = torch.stack(sets)  # shape (B, N_i, 10)
				jet_features = torch.stack(jet_features)
				rave_inputs = torch.stack(rave_inputs)
				with torch.no_grad():
					jet_predictions = model(jet_features,sets,rave_inputs)

			else:
				sets = []
				jet_features = []
				for i in sub_batch:
					sets_i, _, _, jet_features_i, _ = test_data[i]
					sets.append(torch.tensor(sets_i))
					jet_features.append(torch.tensor(jet_features_i))
					
				sets = torch.stack(sets)  # shape (B, N_i, 10)
				jet_features = torch.stack(jet_features)
				with torch.no_grad():
					jet_predictions = model(jet_features,sets)
			predictions += list(jet_predictions.cpu().data.numpy())

	sorted_predictions = [x for _, x in sorted(zip(indx_list, predictions))]

	df = pd.DataFrame(columns=['prediction'])
	df['prediction'] =  sorted_predictions
	df.to_hdf(config.outputfilename+'_predictions.h5',key='df')


if __name__ == '__main__':
	main()
