import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@NETWORK_REGISTRY.register()
class AttentionDeform(nn.Module):
    def __init__(self, channels, head, dim=3, device=None):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channels // head
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.query_filter, self.key_filter, self.value_filter = (
            nn.Conv1d(channels, channels, kernel_size=1).to(device),
            nn.Conv1d(channels, channels, kernel_size=1).to(device),
            nn.Conv1d(channels, channels, kernel_size=1).to(device),
        )
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1).to(device)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        ).to(device)
        self.tran_layer = nn.Linear(channels, dim).to(device)

    def forward(self, eigen):
        # motion1(q) attend to motion(k,v)
        batch_size = eigen.shape[0]
        query, key, value = self.query_filter(eigen).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(eigen).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter(eigen).view(batch_size, self.head, self.head_dim, -1)
        score = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim = -1)
        add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
        add_value = self.mh_filter(add_value)
        motion = eigen + self.cat_filter(torch.cat([eigen, add_value], dim=1))
        motion_new= self.tran_layer(motion.transpose(1,2).squeeze(0)).unsqueeze(0)
        return motion_new
    
