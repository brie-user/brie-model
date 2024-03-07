import logging
from math import log
import time
import datetime
import os
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import test_brie, test_sage, load_data, pos_neg_split, normalize, undersample
from src.utils import filter_feat_and_labels, load_probs, append_extra_features
from src.model import BRIELayer
from src.layers import InterAgg, IntraAgg
from src.graphsage import *


"""
	Training BRIE
	Paper: Enhancing Fraud Detection in GNNs with Synthetic Fraud Generation
           and Integrated Structural Features
    code modified from https://github.com/PonderLY/PC-GNN and from CACO-GNN by Z.Deng and al.
"""


class ModelHandler(object):

    def __init__(self, config):
        args = argparse.Namespace(**config)

        np.random.seed(args.seed)
        random.seed(args.seed)
        # load graph, feature, and label
        [homo, relation1, relation2, relation3], feat_data, labels = load_data(
            args.data_name, prefix=args.data_dir)

        
        indices_rel1, indices_rel2, indices_rel3, labels_rel1, labels_rel2, labels_rel3 = filter_feat_and_labels(
          feat_data, labels, relation1, relation2, relation3)

        # train_test split
        if args.data_name == 'yelp':
            index = list(range(len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=args.train_ratio,
                                                                    random_state=2, shuffle=True)
            idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
                                                                    random_state=2, shuffle=True)

        elif args.data_name == 'amazon':  # amazon
            # 0-3304 are unlabeled nodes
            index = list(range(3305, len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                    train_size=args.train_ratio, random_state=2, shuffle=True)
            idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                    test_size=args.test_ratio, random_state=2, shuffle=True)

        logging.info(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},' +
                     f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}')
        logging.info(f"Classification threshold: {args.thres}")

        # split pos neg sets for under-sampling
        train_pos, train_neg = pos_neg_split(idx_train, y_train)

        # append extra features
        probs_rel1 = load_probs(args.data_name,'rel1', all_feats=False)
        feat_data_rel1 = append_extra_features(feat_data,labels,*probs_rel1)

        probs_rel2 = load_probs(args.data_name,'rel2', all_feats=False)
        feat_data_rel2 = append_extra_features(feat_data,labels,*probs_rel2)

        probs_rel3 = load_probs(args.data_name,'rel3', all_feats=False)
        feat_data_rel3 = append_extra_features(feat_data,labels,*probs_rel3)

        feat_data = normalize(feat_data)
        feat_data_rel1 = normalize(feat_data_rel1)
        feat_data_rel2 = normalize(feat_data_rel2)
        feat_data_rel3 = normalize(feat_data_rel3)

       
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

        # set input graph
        if args.model == 'SAGE' or args.model == 'GCN':
            adj_lists = homo
        else:
            adj_lists = [relation1, relation2, relation3]

        logging.info(
            f'Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}.')

        probs_all_rel1 = load_probs(args.data_name, 'rel1', all_feats=True)
        probs_all_rel2 = load_probs(args.data_name, 'rel2', all_feats=True)
        probs_all_rel3 = load_probs(args.data_name, 'rel3', all_feats=True)

        self.args = args
        self.dataset = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'homo': homo,
                        'feat_data_rel1':feat_data_rel1, 'feat_data_rel2': feat_data_rel2, 'feat_data_rel3': feat_data_rel3,
                        'probs_rel1': probs_all_rel1, 'probs_rel2': probs_all_rel2, 'probs_rel3': probs_all_rel3,
                        'idx_train': idx_train, 'idx_valid': idx_valid, 'idx_test': idx_test,
                        'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
                        'train_pos': train_pos, 'train_neg': train_neg}

    def train(self):
        args = self.args
        feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
        feat_data_rel1, feat_data_rel2 = self.dataset['feat_data_rel1'],  self.dataset['feat_data_rel2']
        feat_data_rel3 = self.dataset['feat_data_rel3']
        probs_rel1, probs_rel2, probs_rel3 = self.dataset['probs_rel1'], self.dataset['probs_rel2'], self.dataset['probs_rel3']
        idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
        idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset[
            'y_valid'], self.dataset['idx_test'], self.dataset['y_test']
        train_pos, train_neg = self.dataset['train_pos'], self.dataset['train_neg']
        
        # initialize model input
        features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
        features.weight = nn.Parameter(
            torch.FloatTensor(feat_data), requires_grad=False)

        features_rel1 = nn.Embedding(feat_data_rel1.shape[0], feat_data_rel1.shape[1])
        features_rel1.weight = nn.Parameter(
            torch.FloatTensor(feat_data_rel1), requires_grad=False)
        
        features_rel2 = nn.Embedding(feat_data_rel2.shape[0], feat_data_rel2.shape[1])
        features_rel2.weight = nn.Parameter(
            torch.FloatTensor(feat_data_rel2), requires_grad=False)

        features_rel3 = nn.Embedding(feat_data_rel3.shape[0], feat_data_rel3.shape[1])
        features_rel3.weight = nn.Parameter(
            torch.FloatTensor(feat_data_rel3), requires_grad=False)

        if args.cuda:
            features.cuda()
            features_rel1.cuda()
            features_rel2.cuda()
            features_rel3.cuda()

        extend_dim = max(feat_data_rel1.shape[1],feat_data_rel2.shape[1],feat_data_rel3.shape[1])   
        # build one-layer models
        if args.model == 'BRIE':
            intra1 = IntraAgg('rel1',features_rel1, feat_data_rel1.shape[1], probs_rel1, args.emb_size,args.contra_dim, train_pos, args.rho, args.temperature, args.data_name, cuda=args.cuda)
            intra2 = IntraAgg('rel2',features_rel2, feat_data_rel2.shape[1], probs_rel2, args.emb_size,args.contra_dim, train_pos, args.rho, args.temperature, args.data_name, cuda=args.cuda)
            intra3 = IntraAgg('rel3',features_rel3, feat_data_rel3.shape[1], probs_rel3, args.emb_size,args.contra_dim, train_pos, args.rho, args.temperature, args.data_name, cuda=args.cuda)
            inter1 = InterAgg(features, feat_data.shape[1], train_pos, args.emb_size, args.contra_dim, extend_dim, adj_lists, [intra1, intra2, intra3], args.rho, inter=args.multi_relation, cuda=args.cuda)
        elif args.model == 'SAGE':
            agg_sage = MeanAggregator(features, cuda=args.cuda)
            enc_sage = Encoder(
                features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, gcn=False, cuda=args.cuda)
        elif args.model == 'GCN':
            agg_gcn = GCNAggregator(features, cuda=args.cuda)
            enc_gcn = GCNEncoder(
                features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, gcn=True, cuda=args.cuda)

        if args.model == 'BRIE':
            gnn_model = BRIELayer(2, inter1, args.lambda_1)
        elif args.model == 'SAGE':
            # the vanilla GraphSAGE model as baseline
            enc_sage.num_samples = 5
            gnn_model = GraphSage(2, enc_sage)
        elif args.model == 'GCN':
            gnn_model = GCN(2, enc_gcn)

        if args.cuda:
            gnn_model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters(
        )), lr=args.lr, weight_decay=args.weight_decay)

        dir_saver = args.save_dir+args.exp_name
        path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, args.model))
        f1_mac_best, auc_best, ep_best = 0, 0, -1

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters(
        )), lr=args.lr, weight_decay=args.weight_decay)
        
        # train the model
        for epoch in range(args.num_epochs):

            sampled_idx_train = undersample(
                train_pos, train_neg, scale=args.undersample)
            random.shuffle(sampled_idx_train)
            num_batches = int(len(sampled_idx_train) / args.batch_size) + 1

            loss = 0.0
            epoch_time = 0

            # mini-batch training
            for batch in range(num_batches):
                start_time = time.time()
                i_start = batch * args.batch_size
                i_end = min((batch + 1) * args.batch_size,
                            len(sampled_idx_train))
                batch_nodes = sampled_idx_train[i_start:i_end]
                batch_label = self.dataset['labels'][np.array(batch_nodes)]
                # print(".count(0)",np.count_nonzero(batch_label == 0))
                # print(".count(1)",np.count_nonzero(batch_label == 1))
                optimizer.zero_grad()
                if args.cuda:
                    loss = gnn_model.loss(batch_nodes, Variable(
                        torch.cuda.LongTensor(batch_label)))
                else:
                    loss = gnn_model.loss(batch_nodes, Variable(
                        torch.LongTensor(batch_label)))
                loss.backward()
                optimizer.step()
                end_time = time.time()
                epoch_time += end_time - start_time
                loss += loss.item()

            logging.info(
                f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')

            # Valid the model for every $valid_epoch$ epoch
            if epoch % args.valid_epochs == 0:
                if args.model == 'SAGE' or args.model == 'GCN':
                    logging.info("Valid at epoch {}".format(epoch))
                    f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_sage(
                        idx_valid, y_valid, gnn_model, args.batch_size, args.thres)
                    if auc_val > auc_best:
                        f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
                        if not os.path.exists(dir_saver):
                            os.makedirs(dir_saver)
                        logging.info('  Saving model ...')
                        torch.save(gnn_model.state_dict(), path_saver)
                else:
                    logging.info("Valid at epoch {}".format(epoch))
                    f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val, recall, fbeta2_1_gnn, fbeta4_1_gnn, recall_gnn1 = test_brie(
                        idx_valid, y_valid, gnn_model, args.batch_size, args.thres)
                    if auc_val > auc_best:
                        f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
                        if not os.path.exists(dir_saver):
                            os.makedirs(dir_saver)
                        logging.info('  Saving model ...')
                        torch.save(gnn_model.state_dict(), path_saver)

        logging.info("Restore model from epoch {}".format(ep_best))
        logging.info("Model path: {}".format(path_saver))
        gnn_model.load_state_dict(torch.load(path_saver))
        if args.model == 'SAGE' or args.model == 'GCN':
            f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_sage(
                idx_test, y_test, gnn_model, args.batch_size, args.thres)
        else:
            logging.debug(f'\ninter1.pos_embs.weight {inter1.pos_embs.weight}')
            f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test, recall, fbeta2_1_gnn, fbeta4_1_gnn, recall_gnn1 = test_brie(
                idx_test, y_test, gnn_model, args.batch_size, args.thres)
        return f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test, recall, fbeta2_1_gnn, fbeta4_1_gnn, recall_gnn1
