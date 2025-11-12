import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@NETWORK_REGISTRY.register()
class DeformNet(nn.Module):
    def __init__(
        self,
        lambda_=10,
        channels=256,
        head=4,
        device=None,
    ):
        nn.Module.__init__(self)
        super(DeformNet, self).__init__()
        self.channels = channels
        self.head = head
        self.min_value = 0.05
        self.max_value = 0.95
        self.lambda_ = lambda_

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.upsample = nn.Conv1d(1, channels, kernel_size=1).to(device)
        self.inject = AttentionPropagation(channels, head, device)
        self.feats_weight = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        ).to(device)
        self.rectify = AttentionPropagation(channels, head, device)

    def forward(
        self,
        feat_x,
        feat_y,
        Pxy,
        vert_x,
        vert_y,
        evecs_x,
        evals_x,
        loc_scores,
        device=None,
    ):

        if device is None:
            device = feat_x.device

        Pxy = Pxy.squeeze()
        vert_F = torch.matmul(Pxy, vert_y) - vert_x
        vert_F = vert_F.squeeze()
        U = evecs_x.squeeze()
        K = torch.diag(evals_x.flatten())
        F = torch.matmul(Pxy, feat_y) - feat_x
        F = F.permute(0, 2, 1)
        feat_x = feat_x.permute(0, 2, 1)
        losc_up = self.upsample(loc_scores.unsqueeze(0).unsqueeze(0))
        feats_prop = self.inject(F, losc_up).to(device)

        P = self.feats_weight(feats_prop).to(device)
        P = torch.clamp(P, self.min_value, self.max_value)
        P_diag = torch.diag_embed(P.squeeze(1)).squeeze()

        _, D, N = feat_x.shape
        sigma2 = torch.sum(F**2) / (N * D)

        W = torch.linalg.solve(
            U.mT @ P_diag @ U + self.lambda_ * sigma2 * K,
            U.mT @ P_diag @ vert_F,
        )

        T = U @ W +vert_x

        return T, P

@NETWORK_REGISTRY.register()
class AttentionPropagation(nn.Module):
    def __init__(self, channels, head, drop=0.1):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channels // head
        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), 
            nn.ReLU(),
            nn.Dropout(p=drop), 
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )

    def forward(self, motion1, motion2):
        # motion1(q) attend to motion(k,v)
        batch_size = motion1.shape[0]
        query, key, value = self.query_filter(motion1).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(motion2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter(motion2).view(batch_size, self.head, self.head_dim, -1)
        score = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim = -1)
        add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
        add_value = self.mh_filter(add_value)
        motion1_new = motion1 + self.cat_filter(torch.cat([motion1, add_value], dim=1))
        return motion1_new
