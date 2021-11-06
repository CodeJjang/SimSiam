import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineHardNegativeMiningTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, mode, margin_ratio=1, pos_ratio=1, neg_pow=1, pos_pow=1, device=None):
        super(OnlineHardNegativeMiningTripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode
        self.margin_ratio = margin_ratio
        self.pos_ratio = pos_ratio
        self.pos_pow = pos_pow
        self.neg_pow = neg_pow
        self.device = device

    def forward(self, emb1, emb2):

        if self.mode == 'Random':
            neg_idx = torch.randint(high=emb1.shape[0], size=(emb1.shape[0],), device=self.device)
            ap_distances = (emb1 - emb2).pow(2).sum(1)
            an_distances = (emb1 - emb2[neg_idx, :]).pow(2).sum(1)
            margin = ap_distances - an_distances

        if (self.mode == 'Hardest') | (self.mode == 'HardPos'):
            sim_matrix = torch.mm(emb1, emb2.transpose(0, 1))
            sim_matrix -= 1000000000 * torch.eye(n=sim_matrix.shape[0], m=sim_matrix.shape[1], device=self.device)
            neg_idx = torch.argmax(sim_matrix, axis=1)  # find negative with highest similarity

        if self.mode == 'Hardest':
            ap_distances = (emb1 - emb2).pow(2).sum(1)
            an_distances = (emb1 - emb2[neg_idx, :]).pow(2).sum(1)

            margin = ap_distances - an_distances

        if self.mode == 'HardPos':
            ap_distances = (emb1 - emb2).pow(2).sum(1)
            an_distances = (emb1 - emb2[neg_idx, :]).pow(2).sum(1)

            # get LARGEST positive distances
            pos_idx = ap_distances.argsort(dim=-1, descending=True)  # sort positive distances
            pos_idx = pos_idx[0:int(self.pos_ratio * pos_idx.shape[0])]  # retain only self.pos_ratio of the positives

            margin = ap_distances[pos_idx] - an_distances[pos_idx]

            # hard examples first: sort margin
            idx = margin.argsort(dim=-1, descending=True)

            # retain a subset of hard examples
            idx = idx[0:int(self.margin_ratio * idx.shape[0])]  # retain some of the examples

            margin = margin[idx]

        losses = F.relu(margin + self.margin)
        idx = torch.where(losses > 0)[0]

        if idx.size()[0] > 0:
            losses = losses[idx].mean()

            if torch.isnan(losses):
                print('Found nan in loss ')
        else:
            losses = 0

        return losses
