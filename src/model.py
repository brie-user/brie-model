import logging
import torch
import torch.nn as nn
from torch.nn import init


"""
    BRIE Model
    Paper: Enhancing Fraud Detection in GNNs with Synthetic Fraud Generation
           and Integrated Structural Features
    code modified from https://github.com/PonderLY/PC-GNN and from CACO-GNN by Z.Deng and al.
"""


class BRIELayer(nn.Module):
    """
    BRIE layer
    """

    def __init__(self, num_classes, inter, lambda_1):
        """Initialize the BRIE model

        Args:
            num_classes: number of classes (2 in our paper)
            inter: the inter-relation aggregator that output the final embedding
            lambda_1: banlance loss (same in paper) 
        """
        super(BRIELayer, self).__init__()
        self.inter = inter
        self.xent = nn.CrossEntropyLoss()
        # the parameter to transform the final embedding
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, inter.embed_dim))
        init.xavier_uniform_(self.weight)
        self.lambda_1 = lambda_1

    def forward(self, nodes, labels, train_flag=True):
        embeds, feat_contra_loss = self.inter(nodes, labels, train_flag)
        scores = self.weight.mm(embeds)
        logging.debug(f"MLP grad {self.weight.grad}")
        return scores.t(), feat_contra_loss

    def to_prob(self, nodes, labels, train_flag=True):
        gnn_logits, __ = self.forward(nodes, labels, train_flag)
        gnn_scores = torch.sigmoid(gnn_logits)
        return gnn_scores

    def loss(self, nodes, labels, train_flag=True):
        gnn_scores, feat_contra_loss = self.forward(nodes, labels, train_flag)
        gnn_loss = self.xent(gnn_scores, labels.squeeze())
        final_loss = gnn_loss + self.lambda_1 * feat_contra_loss
        return final_loss
