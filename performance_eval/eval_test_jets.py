import torch
import uproot
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import metrics
import mpmath

from dataloaders.jets_loader import JetGraphDataset


def _get_rand_index(labels, predictions):
    n_items = len(labels)
    if (n_items < 2):
        return 1
    n_pairs = (n_items * (n_items - 1)) / 2

    correct_pairs = 0
    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if (label_true and pred_true) or ((not label_true) and (not pred_true)):
                correct_pairs += 1

    return correct_pairs / n_pairs


def _error_count(labels, predictions):
    n_items = len(labels)

    true_positives = 0
    false_positive = 0
    false_negative = 0

    for item_i in range(n_items):
        for item_j in range(item_i + 1, n_items):
            label_true = labels[item_i] == labels[item_j]
            pred_true = predictions[item_i] == predictions[item_j]
            if (label_true and pred_true):
                true_positives += 1
            if (not label_true) and pred_true:
                false_positive += 1
            if label_true and (not pred_true):
                false_negative += 1
    return true_positives, false_positive, false_negative


def _get_recall(labels, predictions):
    true_positives, false_positive, false_negative = _error_count(labels, predictions)

    if true_positives + false_negative == 0:
        return 0

    return true_positives / (true_positives + false_negative)


def _get_precision(labels, predictions):
    true_positives, false_positive, false_negative = _error_count(labels, predictions)

    if true_positives + false_positive == 0:
        return 0
    return true_positives / (true_positives + false_positive)


def _f_measure(labels, predictions):
    precision = _get_precision(labels, predictions)
    recall = _get_recall(labels, predictions)

    if precision == 0 or recall == 0:
        return 0

    return 2 * (precision * recall) / (recall + precision)


def eval_jets_on_test_set(model):

    pred = _predict_on_test_set(model)

    test_ds = uproot.open('data/jets_data_all/test/test_data.root')
    jet_df = test_ds['tree'].pandas.df(['jet_flav', 'trk_vtx_index'], flatten=False)
    jet_flav = jet_df['jet_flav']

    target = [x for x in jet_df['trk_vtx_index'].values]

    print('Calculating scores on test set... ', end='')
    start = datetime.now()
    model_scores = {}
    model_scores['RI'] = np.vectorize(_get_rand_index)(target, pred)
    model_scores['ARI'] = np.vectorize(adjustedRI_onesided)(target, pred)
    model_scores['P'] = np.vectorize(_get_precision)(target, pred)
    model_scores['R'] = np.vectorize(_get_recall)(target, pred)
    model_scores['F1'] = np.vectorize(_f_measure)(target, pred)

    end = datetime.now()
    print(f': {str(end - start).split(".")[0]}')

    flavours = {5: 'b jets', 4: 'c jets', 0: 'light jets'}
    metrics_to_table = ['P', 'R', 'F1', 'RI', 'ARI']

    df = pd.DataFrame(index=flavours.values(), columns=metrics_to_table)

    for flav_n, flav in flavours.items():
        for metric in metrics_to_table:
            mean_metric = np.mean(model_scores[metric][jet_flav == flav_n])
            df.at[flav, metric] = mean_metric

    return df


def _predict_on_test_set(model):
    test_ds = JetGraphDataset('test', random_permutation=False)
    model.eval()

    n_tracks = [test_ds[i][0].shape[0] for i in range(len(test_ds))]

    indx_list = []
    predictions = []

    for tracks_in_jet in range(2, np.amax(n_tracks)+1):
        trk_indxs = np.where(np.array(n_tracks) == tracks_in_jet)[0]
        if len(trk_indxs) < 1:
            continue
        indx_list += list(trk_indxs)

        input_batch = torch.stack([test_ds[i][0] for i in trk_indxs])  # shape (B, N_i, 10)


        edge_vals = model(input_batch).squeeze(1)
        edge_scores = 0.5*(edge_vals + edge_vals.transpose(1, 2))
        edge_scores = torch.sigmoid(edge_scores)
        edge_scores[:, np.arange(N), np.arange(N)] = 1.
        
        pred_matrices = compute_clusters_with_partition_score(edge_scores)
        pred_clusters = compute_vertex_assignment(pred_matrices)

        predictions += list(pred_clusters.cpu().data.numpy())  # Shape

    sorted_predictions = [list(x) for _, x in sorted(zip(indx_list, predictions))]
    return sorted_predictions


def compute_partition_score(edge_scores,pred_matrix):

    score = -(pred_matrix*torch.log(edge_scores+0.0000000000001)+(1-pred_matrix)*torch.log(1-edge_scores+0.0000000000001))
    #print(score.shape)
    score = score.sum(dim=1).sum(dim=1)
    return score


def fill_gaps(edge_vals):

    b, n, _ = edge_vals.shape
    
    with torch.no_grad():
        
        pred_matrices = edge_vals.ge(0.5).float() 
        pred_matrices[:, np.arange(n), np.arange(n)] = 1.  # each node is always connected to itself
        ones_now = pred_matrices.sum()
        ones_before = ones_now - 1
        while ones_now != ones_before:  # get connected components - each node connected to all in its component
            ones_before = ones_now
            pred_matrices = torch.matmul(pred_matrices, pred_matrices)
            pred_matrices = pred_matrices.bool().float()  # remain as 0-1 matrices
            ones_now = pred_matrices.sum()

    return pred_matrices

def compute_vertex_assignment(pred_matrix):
    b, n, _ = pred_matrix.shape
    pred_matrix[:, np.arange(n), np.arange(n)] = 1.
    clusters = -1 * torch.ones((b, n))
    tensor_1 = torch.tensor(1.)
    for i in range(n):
        clusters = torch.where(pred_matrix[:, i] == 1, i * tensor_1, clusters)
    
    return clusters.long()

def compute_clusters_with_partition_score(edge_scores):
    # assuming edge_scores has been symmetrized and sigmoid was applied to it
    # edge scores shape is B,N,N
    B,N,_ = edge_scores.shape
    Ne = int(N*(N-1)/2)
    
    r,c = np.triu_indices(N,1)
    r  = np.tile(r,B)
    c = np.tile(c,B)
    z = np.repeat( np.arange(B), Ne)
    flat_edge_scores = edge_scores[z,r,c].view(B,Ne) #shape = B,Ne

    sorted_values, indices = torch.sort(flat_edge_scores,descending=True)
    
    final_edge_decision = torch.zeros(B,N,N)
    final_edge_decision[:, np.arange(N), np.arange(N)] = 1
    flat_sorted_edge_decisions = torch.zeros(B,Ne)
    
    partition_scores = compute_partition_score(edge_scores,final_edge_decision)
    
    for edge_i in range(Ne):
        
        temp_edge_decision = flat_sorted_edge_decisions.clone()
        
        temp_edge_decision[:,edge_i] = torch.where(sorted_values[:,edge_i] > 0.5,torch.tensor(1),torch.tensor(0))
        
        #reverse the sorting,
        temp_edge_decision_unsorted = temp_edge_decision.gather(1, indices.argsort(1))
        temp_partition = torch.zeros(B,N,N)
        temp_partition[z,r,c] = temp_edge_decision_unsorted.flatten()
        temp_partition.transpose(2,1)[z,r,c] = temp_edge_decision_unsorted.flatten()
        temp_partition = fill_gaps(temp_partition)
        
        temp_partition_scores = compute_partition_score(edge_scores,temp_partition)
        
        
        temp_edge_decision[:,edge_i] = torch.where( (temp_partition_scores < partition_scores) & (sorted_values[:,edge_i] > 0.5) ,torch.tensor(1),torch.tensor(0) )
        
        flat_sorted_edge_decisions = temp_edge_decision
        
        temp_edge_decision_unsorted = temp_edge_decision.gather(1, indices.argsort(1))

        final_edge_decision[z,r,c] = temp_edge_decision_unsorted.flatten()
        final_edge_decision.transpose(2,1)[z,r,c] = temp_edge_decision_unsorted.flatten()
        
        partition_scores = compute_partition_score(edge_scores,fill_gaps(final_edge_decision))

    final_edge_decision = fill_gaps(final_edge_decision)
    
    return final_edge_decision



def infer_clusters(edge_vals):
    '''
    Infer the clusters. Enforce symmetry.
    :param edge_vals: predicted edge score values. shape (B, N, N)
    :return: long tensor shape (B, N) of the clusters.
    '''
    # deployment - infer chosen clusters:
    b, n, _ = edge_vals.shape
    with torch.no_grad():
        pred_matrices = edge_vals + edge_vals.transpose(1, 2)  # to make symmetric
        pred_matrices = pred_matrices.ge(0.).float()  # adj matrix - 0 as threshold
        pred_matrices[:, np.arange(n), np.arange(n)] = 1.  # each node is always connected to itself
        ones_now = pred_matrices.sum()
        ones_before = ones_now - 1
        while ones_now != ones_before:  # get connected components - each node connected to all in its component
            ones_before = ones_now
            pred_matrices = torch.matmul(pred_matrices, pred_matrices)
            pred_matrices = pred_matrices.bool().float()  # remain as 0-1 matrices
            ones_now = pred_matrices.sum()

        clusters = -1 * torch.ones((b, n), device=edge_vals.device)
        tensor_1 = torch.tensor(1., device=edge_vals.device)
        for i in range(n):
            clusters = torch.where(pred_matrices[:, i] == 1, i * tensor_1, clusters)

    return clusters.long()

def rand_index(labels,predictions):
    
    n_items = len(labels)
    if(n_items<2):
        return 1
    n_pairs = (n_items*(n_items-1))/2

    correct_pairs = 0
    for item_i in range(n_items):
        for item_j in range(item_i+1,n_items):
            label_true = labels[item_i]==labels[item_j]
            pred_true = predictions[item_i]==predictions[item_j]
            if (label_true and pred_true) or ((not label_true) and (not pred_true)):
                correct_pairs+=1
    
    return correct_pairs/n_pairs


def Expval(labels):
    labels = np.array(labels)
    n = len(labels)
    nchoose2 = (n*(n-1))/2.0
    k = len(list(set(labels)))
    g = [len(np.where(labels==x)[0]) for x in list(set(labels))]
    bn = float( mpmath.bell(n) )
    bnmin1 = float( mpmath.bell(n-1) )
    bnratio = bnmin1/bn
    q = np.sum([(gi*(gi-1))/2 for gi in g])
    return bnratio*(q/nchoose2)+(1-bnratio)*(1-(q/nchoose2))
    
    

def adjustedRI_onesided(labels,predictions):
    
    ri = rand_index(labels,predictions)
    expval = Expval(labels)
    ari = (ri - expval)/(1-expval)
    return ari

def Error_count(labels,predictions):
    n_items = len(labels)

    true_positives = 0
    false_positive = 0
    false_negative = 0
    
    for item_i in range(n_items):
        for item_j in range(item_i+1,n_items):
            label_true = labels[item_i]==labels[item_j]
            pred_true = predictions[item_i]==predictions[item_j]
            if (label_true and pred_true):
                true_positives+=1
            if (not label_true) and pred_true:
                false_positive+=1
            if label_true and (not pred_true):
                false_negative+=1
    return true_positives,false_positive,false_negative

def Recall(labels,predictions):
    true_positives,false_positive,false_negative = Error_count(labels,predictions)
    
    if true_positives+false_negative == 0:
        return 0
    
    return true_positives/(true_positives+false_negative)
    
def Precision(labels,predictions):
    true_positives,false_positive,false_negative = Error_count(labels,predictions)
    
    if true_positives+false_positive == 0:
        return 0
    return true_positives/(true_positives+false_positive)

def f_measure(labels,predictions):
    
    precision = Precision(labels,predictions)
    recall = Recall(labels,predictions)
    
    if precision==0 or recall==0:
        return 0
    
    return 2*(precision*recall)/(recall+precision)
