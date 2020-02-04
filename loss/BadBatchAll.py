import numpy as np
import torch
from torch import nn

class BadBatchAllLoss(nn.Module):
  def __init__(self, margin=0.2):
    super(BadBatchAllLoss, self).__init__()
    self.margin = margin
    self.ranking_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

  def forward(self, outputs, labels):
    running_loss = 0
    triplet_count = 0
    
    for i, anchor in enumerate(labels):
      for j, pos in enumerate(labels):
        if anchor == pos and i != j:
          for g, neg in enumerate(labels):
            if neg != pos:
              running_loss += self.ranking_loss(outputs[i].unsqueeze(0), outputs[j].unsqueeze(0), outputs[g].unsqueeze(0))
              triplet_count += 1
    
    # return average loss for outputs
    return running_loss / triplet_count, 0, 0, 0

  def __str__(self):
      return "Bad Batch All, margin = {}".format(self.margin)
