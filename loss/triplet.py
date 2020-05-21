
import numpy as np
import torch
from torch import nn

class TripletLoss(nn.Module):
   def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

   def forward(self, outputs, labels):
      running_loss = 0
      assert outputs.shape[0] % 3 == 0, "Triplet loss requires outputs to be composed of triplets, but shape was {}".format(outputs.shape)
      assert type(outputs) == torch.Tensor, "Outputs must be a tensor"

      i = 0
      while (i < outputs.shape[0]):
         anchor = outputs[i].unsqueeze(0)
         pos = outputs[i+1].unsqueeze(0)
         neg = outputs[i+2].unsqueeze(0)
         running_loss += self.ranking_loss(anchor, pos, neg)
         i += 3
      
      # return average loss for outputs
      return running_loss / (outputs.shape[0] / 3)

   def __str__(self):
        return "Triplet, margin = {}".format(self.margin)