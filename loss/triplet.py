import numpy as np
import torch

from loss.loss import Loss

class TripletLoss(Loss):
   def __init__(self, margin=1.0):
      super().__init__(torch.nn.TripletMarginLoss(margin=margin, p=2), "Offline Triplet")

   def getLoss(self, batch, labels, device="cpu"):
      running_loss = 0
      assert batch.shape[0] % 3 == 0, "Triplet loss requires batches to be composed of triplets"
      assert type(batch) == torch.Tensor, "Outputs must be a tensor"

      i = 0
      while (i < batch.shape[0]):
         anchor = batch[i].unsqueeze(0)
         pos = batch[i+1].unsqueeze(0)
         neg = batch[i+2].unsqueeze(0)
         running_loss += self.eval(anchor, pos, neg)
         i += 3
      
      # return average loss for batch
      return running_loss / (batch.shape[0] / 3)
