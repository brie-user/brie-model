import logging
import pickle
import random
import os
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, fbeta_score
from collections import defaultdict

import torch


"""
	Utility functions to handle data and evaluate model.
"""


def load_data(data, prefix='data/'):
    """
    Load graph, feature, and label given dataset name
    :returns: home and single-relation graphs, feature, label
    """

    if data == 'yelp':
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)
        file.close()
    elif data == 'amazon':
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)

    return [homo, relation1, relation2, relation3], feat_data, labels

def load_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def filter_feat_and_labels(feat_data, labels, relation1, relation2, relation3):
    def filter_nodes(relation_dict):
        return [node for node, neighbors in relation_dict.items() if len(neighbors) > 1]

    nodes_relation1 = filter_nodes(relation1)
    nodes_relation2 = filter_nodes(relation2)
    nodes_relation3 = filter_nodes(relation3)
   
    indices_relation1 = np.array(nodes_relation1)
    indices_relation2 = np.array(nodes_relation2)
    indices_relation3 = np.array(nodes_relation3)

    labels_relation1 = np.array([labels[node] for node in nodes_relation1])
    labels_relation2 = np.array([labels[node] for node in nodes_relation2])
    labels_relation3 = np.array([labels[node] for node in nodes_relation3])

    return indices_relation1, indices_relation2, indices_relation3, labels_relation1, \
    labels_relation2, labels_relation3

def create_synthetic_node(*distributions):
    synthetic_node = []

    for distribution in distributions:
        distribution_name = distribution[0]
        distribution_values = distribution[1]

        degree_ranges = list(distribution_values.keys())

        probabilities = list(distribution_values.values())

        # Choose a degree range based on the given distribution
        selected_range = random.choices(degree_ranges, probabilities)[0]

        lower_bound, upper_bound = selected_range.split('-')
        lower_bound = float(lower_bound)

        if upper_bound == 'inf':
            synthetic_value = random.uniform(lower_bound, lower_bound + 100)  # Choose a high value within the range
        else:
            upper_bound = float(upper_bound)
            synthetic_value = random.uniform(lower_bound, upper_bound)

        # Add the chosen degree and distribution type to the synthetic node
        synthetic_node.append(synthetic_value)

    synthetic_node_array = np.array(synthetic_node)
    synthetic_node_array = normalize_array(synthetic_node_array)

    return synthetic_node_array


def append_extra_features(feat_data, labels, *distributions_pairs):

  extended_feat_data = np.zeros((feat_data.shape[0], feat_data.shape[1] + len(distributions_pairs)))
  extended_feat_data[:, :feat_data.shape[1]] = feat_data

  for row_index, row in enumerate(feat_data):
    new_row = row

    for distribution_pair in distributions_pairs:
      #print("distribution_pair",distribution_pair)
      if labels[row_index] == 0:
        distribution = distribution_pair[0]
      elif labels[row_index] == 1:
        distribution = distribution_pair[1]

      degree_ranges = list(distribution.keys())
      probabilities = list(distribution.values())

      # Choose a degree range based on the given distribution
      selected_range = random.choices(degree_ranges, probabilities)[0]

      lower_bound, upper_bound = selected_range.split('-')
      lower_bound = float(lower_bound)

      if upper_bound == 'inf':
          synthetic_value = random.uniform(lower_bound, lower_bound + 100)  # Choose a high value within the range
      else:
          upper_bound = float(upper_bound)
          synthetic_value = random.uniform(lower_bound, upper_bound)
      
      new_row = np.append(new_row,synthetic_value)

    extended_feat_data[row_index] = new_row
    
  return extended_feat_data

def load_probs(data, relation, all_feats=False):
  prefix = 'data/prob_dists/'

  prob_distrs=[]

  if data == 'yelp':
    if all_feats==True:
      prob_cols_list_yelp = load_from_pickle(os.path.join(prefix, 'prob_cols_list_yelp.pkl'))
      for i, prob_col in enumerate(prob_cols_list_yelp):
        prob_distrs.append(prob_col)

    if relation=='rel1':
      prob_degree_rel1_yelp = load_from_pickle(os.path.join(prefix, 'prob_degree_rel1_yelp.pkl'))
      prob_opposite_rel1_yelp = load_from_pickle(os.path.join(prefix, 'prob_opposite_rel1_yelp.pkl'))
      prob_pref_attachment_rel1_yelp = load_from_pickle(os.path.join(prefix, 'prob_pref_attachment_rel1_yelp.pkl'))
      prob_clustering_coef_rel1_yelp = load_from_pickle(os.path.join(prefix, 'prob_clustering_coef_rel1_yelp.pkl'))
      prob_struct_sim_rel1_yelp = load_from_pickle(os.path.join(prefix, 'prob_struct_sim_rel1_yelp.pkl'))
      
      prob_distrs.extend([prob_degree_rel1_yelp, prob_opposite_rel1_yelp, prob_pref_attachment_rel1_yelp, \
      prob_clustering_coef_rel1_yelp, prob_struct_sim_rel1_yelp])
      
    if relation=='rel2':
      prob_degree_rel2_yelp = load_from_pickle(os.path.join(prefix, 'prob_degree_rel2_yelp.pkl'))
      prob_second_degree_rel2_yelp = load_from_pickle(os.path.join(prefix, 'prob_second_degree_rel2_yelp.pkl'))
      prob_opposite_rel2_yelp = load_from_pickle(os.path.join(prefix, 'prob_opposite_rel2_yelp.pkl'))
      prob_pref_attachment_rel2_yelp = load_from_pickle(os.path.join(prefix, 'prob_pref_attachment_rel2_yelp.pkl'))
      prob_clustering_coef_rel2_yelp = load_from_pickle(os.path.join(prefix, 'prob_clustering_coef_rel2_yelp.pkl'))
      prob_struct_sim_rel2_yelp = load_from_pickle(os.path.join(prefix, 'prob_struct_sim_rel2_yelp.pkl'))
      
      prob_distrs.extend([prob_degree_rel2_yelp, prob_second_degree_rel2_yelp, prob_opposite_rel2_yelp, \
      prob_pref_attachment_rel2_yelp, prob_clustering_coef_rel2_yelp, prob_struct_sim_rel2_yelp])

    if relation=='rel3':
      prob_degree_rel3_yelp = load_from_pickle(os.path.join(prefix, 'prob_degree_rel3_yelp.pkl'))
      prob_opposite_rel3_yelp = load_from_pickle(os.path.join(prefix, 'prob_opposite_rel3_yelp.pkl'))
      prob_pref_attachment_rel3_yelp = load_from_pickle(os.path.join(prefix, 'prob_pref_attachment_rel3_yelp.pkl'))
      prob_struct_sim_rel3_yelp = load_from_pickle(os.path.join(prefix, 'prob_struct_sim_rel3_yelp.pkl'))
      
      prob_distrs.extend([prob_degree_rel3_yelp, prob_opposite_rel3_yelp, prob_pref_attachment_rel3_yelp, prob_struct_sim_rel3_yelp])
  
  elif data=='amazon':
    if all_feats==True:
      prob_cols_list_amz = load_from_pickle(os.path.join(prefix, 'prob_cols_list_amz.pkl'))
      for i, prob_col in enumerate(prob_cols_list_amz):
        prob_distrs.append(prob_col)
    
    if relation=='rel1':
      prob_degree_rel1_amz = load_from_pickle(os.path.join(prefix, 'prob_degree_rel1_amz.pkl'))
      prob_second_degree_rel1_amz = load_from_pickle(os.path.join(prefix, 'prob_second_degree_rel1_amz.pkl'))
      prob_opposite_rel1_amz = load_from_pickle(os.path.join(prefix, 'prob_opposite_rel1_amz.pkl'))
      prob_pref_attachment_rel1_amz = load_from_pickle(os.path.join(prefix, 'prob_pref_attachment_rel1_amz.pkl'))
      prob_clustering_coef_rel1_amz = load_from_pickle(os.path.join(prefix, 'prob_clustering_coef_rel1_amz.pkl'))
      prob_struct_sim_rel1_amz = load_from_pickle(os.path.join(prefix, 'prob_struct_sim_rel1_amz.pkl'))

      prob_distrs.extend([prob_degree_rel1_amz, prob_second_degree_rel1_amz, prob_opposite_rel1_amz, \
      prob_pref_attachment_rel1_amz, prob_clustering_coef_rel1_amz, prob_struct_sim_rel1_amz])

    if relation=='rel2':
      prob_degree_rel2_amz = load_from_pickle(os.path.join(prefix, 'prob_degree_rel2_amz.pkl'))
      prob_second_degree_rel2_amz = load_from_pickle(os.path.join(prefix, 'prob_second_degree_rel2_amz.pkl'))
      prob_opposite_rel2_amz = load_from_pickle(os.path.join(prefix, 'prob_opposite_rel2_amz.pkl'))
      prob_pref_attachment_rel2_amz = load_from_pickle(os.path.join(prefix, 'prob_pref_attachment_rel2_amz.pkl'))
      prob_clustering_coef_rel2_amz = load_from_pickle(os.path.join(prefix, 'prob_clustering_coef_rel2_amz.pkl'))
      prob_struct_sim_rel2_amz = load_from_pickle(os.path.join(prefix, 'prob_struct_sim_rel2_amz.pkl'))
      
      prob_distrs.extend([prob_degree_rel2_amz, prob_second_degree_rel2_amz, prob_opposite_rel2_amz, \
      prob_pref_attachment_rel2_amz, prob_clustering_coef_rel2_amz, prob_struct_sim_rel2_amz])
    
    if relation=='rel3':
      prob_degree_rel3_amz = load_from_pickle(os.path.join(prefix, 'prob_degree_rel3_amz.pkl'))
      prob_second_degree_rel3_amz = load_from_pickle(os.path.join(prefix, 'prob_second_degree_rel3_amz.pkl'))
      prob_opposite_rel3_amz = load_from_pickle(os.path.join(prefix, 'prob_opposite_rel3_amz.pkl'))
      prob_pref_attachment_rel3_amz = load_from_pickle(os.path.join(prefix, 'prob_pref_attachment_rel3_amz.pkl'))
      prob_clustering_coef_rel3_amz = load_from_pickle(os.path.join(prefix, 'prob_clustering_coef_rel3_amz.pkl'))
      prob_struct_sim_rel3_amz = load_from_pickle(os.path.join(prefix, 'prob_struct_sim_rel3_amz.pkl'))
      
      prob_distrs.extend([prob_degree_rel3_amz, prob_second_degree_rel3_amz, prob_opposite_rel3_amz, \
      prob_pref_attachment_rel3_amz, prob_clustering_coef_rel3_amz, prob_struct_sim_rel3_amz])

  return prob_distrs


def normalize(mx):
    """
            Row-normalize sparse matrix
            Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_array(arr):
    """
    Normalize a NumPy array so that its elements sum to 1.
    """
    total = np.sum(arr)
    if total == 0:
        raise ValueError("Cannot normalize an array with a sum of 0.")
    normalized_arr = arr / total
    return normalized_arr

def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def pos_neg_split(nodes, labels):
    """
    Find positive and negative nodes given a list of nodes and their labels
    :param nodes: a list of nodes
    :param labels: a list of node labels
    :returns: the spited positive and negative nodes
    """
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])

    return pos_nodes, neg_nodes


def pick_step(idx_train, y_train, adj_list, size):
    degree_train = [len(adj_list[node]) for node in idx_train]
    lf_train = (y_train.sum()-len(y_train))*y_train + len(y_train)
    smp_prob = np.array(degree_train) / lf_train
    return random.choices(idx_train, weights=smp_prob, k=size)


def test_sage(test_cases, labels, model, batch_size, thres=0.5):
    """
    Test the performance of GraphSAGE
    :param test_cases: a list of testing node
    :param labels: a list of testing node labels
    :param model: the GNN model
    :param batch_size: number nodes in a batch
    """

    test_batch_num = int(len(test_cases) / batch_size) + 1
    gnn_pred_list = []
    gnn_prob_list = []
    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob = model.to_prob(batch_nodes)

        gnn_prob_arr = gnn_prob.data.cpu().numpy()[:, 1]
        gnn_pred = prob2pred(gnn_prob_arr, thres)

        gnn_pred_list.extend(gnn_pred.tolist())
        gnn_prob_list.extend(gnn_prob_arr.tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
    f1_binary_1_gnn = f1_score(labels, np.array(
        gnn_pred_list), pos_label=1, average='binary')
    f1_binary_0_gnn = f1_score(labels, np.array(
        gnn_pred_list), pos_label=0, average='binary')
    f1_micro_gnn = f1_score(labels, np.array(gnn_pred_list), average='micro')
    f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average='macro')
    conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
    tn, fp, fn, tp = conf_gnn.ravel()
    gmean_gnn = conf_gmean(conf_gnn)

    logging.info(f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}" +
                 f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}")
    logging.info(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
    return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn, gmean_gnn


def prob2pred(y_prob, thres=0.5):
    """
    Convert probability to predicted results according to given threshold
    :param y_prob: numpy array of probability in [0, 1]
    :param thres: binary classification threshold, default 0.5
    :returns: the predicted result with the same shape as y_prob
    """
    y_pred = np.zeros_like(y_prob, dtype=np.int32)
    y_pred[y_prob >= thres] = 1
    y_pred[y_prob < thres] = 0
    return y_pred


def test_brie(test_cases, labels, model, batch_size, thres=0.5):
    """
    Test the performance of BRIE and its variants
    :param test_cases: a list of testing node
    :param labels: a list of testing node labels
    :param model: the GNN model
    :param batch_size: number nodes in a batch
    :returns: the AUC and Recall of GNN and Simi modules
    """

    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1_gnn = 0.0
    acc_gnn = 0.0
    recall_gnn = 0.0
    f1_label1 = 0.0
    acc_label1 = 0.00
    recall_label1 = 0.0
    gnn_pred_list = []
    gnn_prob_list = []
    label_list1 = []
    recall_gnn1 = 0.0
    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob = model.to_prob(batch_nodes, batch_label, train_flag=False)

        gnn_prob_arr = gnn_prob.data.cpu().numpy()[:, 1]
        gnn_pred = prob2pred(gnn_prob_arr, thres)

        recall_gnn += recall_score(batch_label,
                                   gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")

        gnn_pred_list.extend(gnn_pred.tolist())
        gnn_prob_list.extend(gnn_prob_arr.tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
    ap_gnn = average_precision_score(labels, np.array(gnn_prob_list))
    f1_binary_1_gnn = f1_score(labels, np.array(
        gnn_pred_list), pos_label=1, average='binary')
    f1_binary_0_gnn = f1_score(labels, np.array(
        gnn_pred_list), pos_label=0, average='binary')
    f1_micro_gnn = f1_score(labels, np.array(gnn_pred_list), average='micro')
    f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average='macro')
    conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
    fbeta2_1_gnn = fbeta_score(labels, np.array(
        gnn_pred_list), pos_label=1, beta=2, average='binary')
    fbeta4_1_gnn = fbeta_score(labels, np.array(
        gnn_pred_list), pos_label=1, beta=4, average='binary')
    tn, fp, fn, tp = conf_gnn.ravel()
    gmean_gnn = conf_gmean(conf_gnn)
    recall = recall_gnn / test_batch_num
    recall_gnn1 = recall_score(labels, np.array(
        gnn_pred_list), pos_label=1, average='binary')
    logging.info(f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}" +
                 f"\tF1-macro: {f1_macro_gnn:.4f}\tF1-micro: {f1_micro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}\tRecall: {recall:.4f}")
    logging.info(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
    logging.info(
        f"   fbeta2_1: {fbeta2_1_gnn}\tfbeta4_1: {fbeta4_1_gnn} \trecall_gnn1: {recall_gnn1}")

    return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn, gmean_gnn, recall, fbeta2_1_gnn, fbeta4_1_gnn, recall_gnn1


def conf_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp*tn/((tp+fn)*(tn+fp)))**0.5


def compute_pos_cosine_similarity_list(pos_emb, features):
    sim_list = [torch.cosine_similarity(pos_emb, feat, dim=1) if len(
        feat.shape) > 1 else torch.cosine_similarity(pos_emb, feat.unsqueeze(0), dim=1) for feat in features]
    return sim_list


def compute_cosine_similarity_list(center_features, r_features):
    sim_list = [torch.cosine_similarity(center_features[idx].unsqueeze(0), feat, dim=1) if len(feat.shape) > 1 else torch.cosine_similarity(
        center_features[idx].unsqueeze(0), feat.unsqueeze(0), dim=1) for idx, feat in enumerate(r_features)]
    return sim_list


def compute_cosine_similarity(pos_emb, features):
    return torch.cosine_similarity(pos_emb, features, dim=1)


def undersample(pos_nodes, neg_nodes, scale=1):
    """
    Under-sample the negative nodes
    :param pos_nodes: a list of positive nodes
    :param neg_nodes: a list negative nodes
    :param scale: the under-sampling scale
    :return: a list of under-sampled batch nodes
    """

    aux_nodes = cp.deepcopy(neg_nodes)
    aux_nodes = random.sample(aux_nodes, k=int(len(pos_nodes)*scale))
    batch_nodes = pos_nodes + aux_nodes

    return batch_nodes
