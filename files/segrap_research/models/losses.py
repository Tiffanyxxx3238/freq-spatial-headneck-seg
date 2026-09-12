"""
Loss functions for AFS-DSN training.
CombinedLoss = CrossEntropy + Dice (original from 02_model.ipynb).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def dice_loss(self, pred, target):
        smooth = 1e-5
        pred = F.softmax(pred, dim=1)
        toh = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        inter = (pred * toh).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + toh.sum(dim=(2, 3, 4))
        dice = (2.0 * inter + smooth) / (union + smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        if isinstance(pred, dict):
            pred = pred['output']
        return self.ce(pred, target) + self.dice_loss(pred, target)
