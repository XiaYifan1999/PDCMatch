import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class DeformLoss(nn.Module):
    def __init__(self, label="full", weight=10.0):
        super().__init__()
        if label == "noP":
            self.ty = 1
        elif label == "noOutlier":
            self.ty = 2
        elif label == "full":
            self.ty = 3
        else:
            raise ValueError(f"Unknown label: {label}")
        self.weight = weight

    def forward(self, a, b, P):

        P_extend = P.unsqueeze(0)
        diffs = torch.sum(torch.abs(a - b) ** 2, dim=-1)
        loss = torch.sum(torch.sqrt(diffs) * P_extend)

        out_weight = a.size(1) - torch.sum(P)

        if self.ty == 1:
            deformloss = loss
        elif self.ty == 2:
            deformloss = out_weight / torch.sum(P)
        elif self.ty == 3:
            deformloss = loss * out_weight / torch.sum(P)

        return self.weight * deformloss

@LOSS_REGISTRY.register()
class CosineLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.weight = loss_weight
    
    def forward(self, Fx, Fy, Pxy):

        cos_sim = F.cosine_similarity(Pxy @ Fy, Fx, dim=-1)
        feat_dif = torch.mean(1-cos_sim)

        return self.weight * feat_dif
    
@LOSS_REGISTRY.register()
class SmoothLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.weight = loss_weight
    
    def forward(self, Target, Transform, ann_weight=1.0):
        
        diffs = torch.sum(torch.abs(Target-Transform)** 2, dim=-1)
        # loss = torch.mean(torch.sqrt(diffs)) # /torch.sum(loc_scores)
        loss = torch.mean(torch.sqrt(diffs))
        # loss = (loc_scores.size(0)-torch.sum(torch.sqrt(diffs)))/torch.sum(loc_scores)
        return self.weight * loss*ann_weight
    
@LOSS_REGISTRY.register()
class LocLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.weight = loss_weight
    
    def forward(self, loc_scores, ann_weight=1.0):

        loss = (loc_scores.size(0)-torch.sum(loc_scores))/torch.sum(loc_scores)
        return self.weight * loss*ann_weight
    
    