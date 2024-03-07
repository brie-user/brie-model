import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from src.losses import ContrastiveLoss
import numpy as np
import random
from operator import itemgetter
import math
from src.utils import compute_cosine_similarity, compute_cosine_similarity_list, compute_pos_cosine_similarity_list

"""
    BRIE Layers
	Paper: Enhancing Fraud Detection in GNNs with Synthetic Fraud Generation
           and Integrated Structural Features
    code modified from https://github.com/PonderLY/PC-GNN and from CACO-GNN by Z.Deng and al.
"""


class InterAgg(nn.Module):

    def __init__(self, features, feature_dim, train_pos,
                 embed_dim, contra_dim, extend_dim, adj_lists, intraggs, rho,
                 inter='GNN', cuda=True):
        """
        Initialize the inter-relation aggregator
        Args:
            features: the input node features or embeddings for all nodes
            feature_dim: the input dimension
            train_pos: positive nodes in train set
            embed_dim: the output dimension
            contra_dim: contrastive embedding dim
            adj_lists: a list of adjacency lists for each single-relation graph
            intraggs: the intra-relation aggregators used by each single-relation graph
            rho: thresholds for sampling
            inter: the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'. Defaults to 'GNN'.
            cuda: whether to use GPU. Defaults to True.
        """        
        super(InterAgg, self).__init__()

        self.features = features
        self.adj_lists = adj_lists
        self.intra_agg1 = intraggs[0]
        self.intra_agg2 = intraggs[1]
        self.intra_agg3 = intraggs[2]
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.extend_dim = extend_dim
        self.inter = inter
        self.cuda = cuda
        self.intra_agg1.cuda = cuda
        self.intra_agg2.cuda = cuda
        self.intra_agg3.cuda = cuda
        self.train_pos = train_pos
        self.contra_dim = contra_dim

        # initial filtering thresholds
        self.thresholds = [rho]*3

        # parameter used to transform node embeddings before inter-relation aggregation
        self.weight = nn.Parameter(torch.FloatTensor(
            self.embed_dim*len(intraggs)+self.feat_dim, self.embed_dim))
        init.xavier_uniform_(self.weight)
        self.pos_embs = nn.Embedding(3, self.contra_dim)
        self.rsimTrans = nn.Parameter(
          torch.FloatTensor(self.extend_dim, self.contra_dim))
        init.xavier_uniform_(self.rsimTrans)

    def forward(self, nodes, labels, train_flag=True):
        """
        :param nodes: a list of batch node ids
        :param labels: a list of batch node labels
        :param train_flag: indicates whether in training or testing mode
        :return combined: the embeddings of a batch of input node features
        :return center_scores: the label-aware scores of batch nodes
        """

        # extract 1-hop neighbor ids from adj lists of each single-relation graph
        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        # find unique nodes and their neighbors used in current batch
        unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
                                 set.union(*to_neighs[2], set(nodes)))

        # get neighbor node id list for each batch node and relation
        r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
        r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
        r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

        if self.cuda:
            batch_features = self.features(
                torch.cuda.LongTensor(list(unique_nodes)))
            pos_features = self.features(
                torch.cuda.LongTensor(list(self.train_pos)))
        else:
            batch_features = self.features(
                torch.LongTensor(list(unique_nodes)))
            pos_features = self.features(
                torch.LongTensor(list(self.train_pos)))

        id_mapping = {node_id: index for node_id, index in zip(
            unique_nodes, range(len(unique_nodes)))}

        sim_batch_features = F.relu(batch_features.mm(self.rsimTrans[:self.feat_dim]))
        sim_pos_features = F.relu(pos_features.mm(self.rsimTrans[:self.feat_dim]))
        r1_features = [sim_batch_features[itemgetter(
            *to_neigh)(id_mapping), :] for to_neigh in r1_list]
        r2_features = [sim_batch_features[itemgetter(
            *to_neigh)(id_mapping), :] for to_neigh in r2_list]
        r3_features = [sim_batch_features[itemgetter(
            *to_neigh)(id_mapping), :] for to_neigh in r3_list]

        center_features = sim_batch_features[itemgetter(*nodes)(id_mapping), :]
        center_origin_features = batch_features[itemgetter(
            *nodes)(id_mapping), :]

        r1_origin_features = [batch_features[itemgetter(
            *to_neigh)(id_mapping), :] for to_neigh in r1_list]
        r2_origin_features = [batch_features[itemgetter(
            *to_neigh)(id_mapping), :] for to_neigh in r2_list]
        r3_origin_features = [batch_features[itemgetter(
            *to_neigh)(id_mapping), :] for to_neigh in r3_list]

        if self.cuda:
            pos_emb1 = self.pos_embs(torch.LongTensor([0]).cuda())
            pos_emb2 = self.pos_embs(torch.LongTensor([1]).cuda())
            pos_emb3 = self.pos_embs(torch.LongTensor([2]).cuda())
        else:
            pos_emb1 = self.pos_embs(torch.LongTensor([0]))
            pos_emb2 = self.pos_embs(torch.LongTensor([1]))
            pos_emb3 = self.pos_embs(torch.LongTensor([2]))

        r1_neigh_scores = compute_cosine_similarity_list(
            center_origin_features, r1_origin_features)
        r2_neigh_scores = compute_cosine_similarity_list(
            center_origin_features, r2_origin_features)
        r3_neigh_scores = compute_cosine_similarity_list(
            center_origin_features, r3_origin_features)

        r1_neigh_pos_scores = compute_pos_cosine_similarity_list(
            pos_emb1, r1_features)
        r2_neigh_pos_scores = compute_pos_cosine_similarity_list(
            pos_emb2, r2_features)
        r3_neigh_pos_scores = compute_pos_cosine_similarity_list(
            pos_emb3, r3_features)

        r1_pos_scores = compute_cosine_similarity(pos_emb1, sim_pos_features)
        r2_pos_scores = compute_cosine_similarity(pos_emb2, sim_pos_features)
        r3_pos_scores = compute_cosine_similarity(pos_emb3, sim_pos_features)

        r1_center_scores = compute_cosine_similarity(pos_emb1, center_features)
        r2_center_scores = compute_cosine_similarity(pos_emb2, center_features)
        r3_center_scores = compute_cosine_similarity(pos_emb3, center_features)

        # count the number of neighbors kept for aggregation for each batch node and relation
        r1_sample_num_list = [
            math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
        r2_sample_num_list = [
            math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
        r3_sample_num_list = [
            math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

        # intra-aggregation steps for each relation
        r1_feats, r1_loss = self.intra_agg1.forward(nodes, labels, r1_list,  r1_sample_num_list, r1_center_scores,
                                                    r1_neigh_scores, r1_neigh_pos_scores, r1_pos_scores, pos_emb1, self.rsimTrans, train_flag)
        r2_feats, r2_loss = self.intra_agg2.forward(nodes, labels, r2_list,  r2_sample_num_list, r2_center_scores,
                                                    r2_neigh_scores, r2_neigh_pos_scores, r2_pos_scores, pos_emb2, self.rsimTrans, train_flag)
        r3_feats, r3_loss = self.intra_agg3.forward(nodes, labels, r3_list,  r3_sample_num_list, r3_center_scores,
                                                    r3_neigh_scores, r3_neigh_pos_scores, r3_pos_scores, pos_emb3, self.rsimTrans, train_flag)

        # get features or embeddings for batch nodes
        if self.cuda and isinstance(nodes, list):
            index = torch.LongTensor(nodes).cuda()
        else:
            index = torch.LongTensor(nodes)

        self_feats = self.features(index)


        # concat the intra-aggregated embeddings from each relation 
        cat_feats = torch.cat(
            (self_feats, r1_feats, r2_feats, r3_feats), dim=1)

        combined = F.relu(cat_feats.mm(self.weight).t())
        feat_contra_loss = (r1_loss+r2_loss+r3_loss)/3
        
        return combined, feat_contra_loss


class IntraAgg(nn.Module):

    def __init__(self, rel_name, features, feat_dim, probs_rel, emb_size, contra_dim, train_pos, rho, temperature, data_name, cuda=False):
        """
        Initialize the intra-relation aggregator
        :param features: the input node features or embeddings for all nodes
        :param feat_dim: the input dimension
        :train_pos: positive nodes in training set
        :param cuda: whether to use GPU
        """
        super(IntraAgg, self).__init__()

        self.rel_name = rel_name
        self.features = features
        self.cuda = cuda
        self.feat_dim = feat_dim
        self.probs_rel = probs_rel
        self.emb_size = emb_size
        self.contra_dim = contra_dim
        self.train_pos = train_pos
        self.rho = rho
        self.data_name = data_name
        self.ConLoss = ContrastiveLoss(
            temperature=temperature, base_temperature=temperature)
        self.weight = nn.Parameter(
            torch.FloatTensor(2*self.feat_dim+self.contra_dim, self.emb_size))
        self.diff = nn.Parameter(torch.FloatTensor(self.emb_size, 1))
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.diff)

    def forward(self, nodes, batch_labels, to_neighs_list, sample_list, batch_scores, neigh_scores, neigh_pos_scores, pos_scores, pos_emb, simTrans, train_flag):
        """
        Code partially from https://github.com/williamleif/graphsage-simple/
        :param nodes: list of nodes in a batch
        :param to_neighs_list: neighbor node id list for each batch node in one relation
        :param batch_scores: the label-aware scores of batch nodes
        :param neigh_scores: the label-aware scores of 1-hop neighbors each batch node in one relation
        :param pos_scores: the label-aware scores 1-hop neighbors for the minority positive nodes
        :param train_flag: indicates whether in training or testing mode
        :param sample_list: the number of neighbors kept for each batch node in one relation
        :return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
        :return samp_scores: the average neighbor distances for each relation after filtering
        """

        samp_neighs = sample_cosistent_neibors(
            nodes, batch_scores, neigh_scores, neigh_pos_scores, to_neighs_list, sample_list)
        if isinstance(batch_labels, np.ndarray):
            if self.cuda:
                batch_labels = torch.FloatTensor(batch_labels).cuda()
            else:
                batch_labels = torch.FloatTensor(batch_labels)

        # find the unique nodes among batch nodes and the filtered neighbors
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # intra-relation aggregation only with sampled neighbors
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n]
                          for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs))
                       for _ in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()

        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)  # mean aggregator

        if self.cuda:
            self_feats = self.features(torch.LongTensor(nodes).cuda())
            embed_matrix = self.features(
                torch.LongTensor(unique_nodes_list).cuda())
        else:
            self_feats = self.features(torch.LongTensor(nodes))
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        if train_flag:
          extra_fraud_nodes = (np.count_nonzero(batch_labels == 0)-np.count_nonzero(batch_labels == 1))
          synthetic_fraud_data = synthesize_fraud_nodes(extra_fraud_nodes, self.probs_rel)
          synthetic_data_labels = torch.ones(extra_fraud_nodes, dtype=torch.float32)
          result_feats = torch.cat((self_feats, synthetic_fraud_data), dim=0)
          ext_batch_labels = torch.cat((batch_labels, synthetic_data_labels), dim=0)

          if self.data_name == 'yelp':
            if self.rel_name == 'rel1':
              ext_sim_matrix = F.relu(result_feats.mm(torch.cat((simTrans[:33], simTrans[34:]), dim=0)))
              sim_matrix = F.relu(self_feats.mm(torch.cat((simTrans[:33], simTrans[34:]), dim=0)))
            elif self.rel_name =='rel2':
              ext_sim_matrix = F.relu(result_feats.mm(simTrans))
              sim_matrix = F.relu(self_feats.mm(simTrans))
            elif self.rel_name =='rel3':
              ext_sim_matrix = F.relu(result_feats.mm(torch.cat((simTrans[:33], simTrans[34:36], simTrans[37:]), dim=0)))
              sim_matrix = F.relu(self_feats.mm(torch.cat((simTrans[:33], simTrans[34:36], simTrans[37:]), dim=0)))
          elif self.data_name == 'amazon':
            ext_sim_matrix = F.relu(result_feats.mm(simTrans))
            sim_matrix = F.relu(self_feats.mm(simTrans))
          loss = self.ConLoss.forward(ext_sim_matrix, ext_batch_labels, pos_emb)
        else:
          if self.data_name == 'yelp':
            if self.rel_name == 'rel1':
              sim_matrix = F.relu(self_feats.mm(torch.cat((simTrans[:33], simTrans[34:]), dim=0)))
            elif self.rel_name =='rel2':
              sim_matrix = F.relu(self_feats.mm(simTrans))
            elif self.rel_name =='rel3':
              sim_matrix = F.relu(self_feats.mm(torch.cat((simTrans[:33], simTrans[34:36], simTrans[37:]), dim=0)))
          elif self.data_name == 'amazon':
            sim_matrix = F.relu(self_feats.mm(simTrans))
          loss = self.ConLoss.forward(sim_matrix, batch_labels, pos_emb)
          
        
        to_feats = mask.mm(embed_matrix)
        #concate contrastive embedding and features
        cat_feats=torch.cat((self_feats,to_feats,sim_matrix.detach()),dim=1)
        to_feats = F.relu(cat_feats.mm(self.weight))
        return to_feats, loss

def synthesize_fraud_nodes(num_nodes, probs_rel):

  synthetic_fraud_data = np.zeros((num_nodes, len(probs_rel)))

  for distribution_pair in probs_rel:

    distribution = distribution_pair[1]
    degree_ranges = list(distribution.keys())
    probabilities = list(distribution.values())

    # choose a degree range based on the given distribution
    selected_range = random.choices(degree_ranges, probabilities)[0]

    lower_bound, upper_bound = selected_range.split('-')
    lower_bound = float(lower_bound)

    if upper_bound == 'inf':
        synthetic_value = random.uniform(lower_bound, lower_bound*2)  # Choose a high value within the range
    else:
        upper_bound = float(upper_bound)
        synthetic_value = random.uniform(lower_bound, upper_bound)
  
  return torch.LongTensor(synthetic_fraud_data)

def sample_cosistent_neibors(nodes, center_scores, neigh_scores, neigh_pos_scores, neighs_list, sample_list):
    """
    Filter neighbors according label predictor result with adaptive thresholds
    :param center_scores: the label-aware scores of batch nodes
    :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
    :param neighs_list: neighbor node id list for each batch node in one relation
    :param sample_list: the number of neighbors kept for each batch node in one relation
    :return samp_neighs: the neighbor indices and neighbor simi scores
    :return samp_scores: the average neighbor distances for each relation after filtering
    """

    samp_neighs = []
    for idx, center_score in enumerate(center_scores):
        nodeid = nodes[idx]
        center_score = center_scores[idx]
        neigh_score = neigh_scores[idx]
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]
        neigh_pos_score = neigh_pos_scores[idx]
        diff = torch.abs(neigh_pos_score-center_score)
        sorted_scores1, sorted_indices1 = torch.sort(neigh_score, dim=0, descending=True)
        sorted_scores2, sorted_indices2 = torch.sort(diff, dim=0, descending=False)
        selected_indices1 = sorted_indices1.tolist()
        selected_indices2 = sorted_indices2.tolist()
        # top-p sampling according to distance ranking and thresholds
        if len(neigh_scores[idx]) > num_sample + 1:
            # consistent feature sampling
            selected_neighs = [neighs_indices[n]
                               for n in selected_indices1[:num_sample+1]]
            # consistent contrastive similarity sampling                   
            selected_neighs.extend([neighs_indices[n]
                                   for n in selected_indices2[:num_sample+1]])
        else:
            selected_neighs = neighs_indices
        selected_neighs = set(selected_neighs)
        if(len(selected_neighs) > 1):
            selected_neighs.remove(nodeid)
        samp_neighs.append(selected_neighs)

    return samp_neighs