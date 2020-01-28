import numpy as np
import torch

from loss.loss import Loss

class BatchHardLoss(Loss):
  def __init__(self, margin=1.0):
    super().__init__(torch.nn.TripletMarginLoss(margin=margin, p=2), "Batch Hard")

  def getLoss(self, batch, labels, device="cpu"):
    running_loss = 0
    assert type(batch) == torch.Tensor, "Outputs must be a tensor"

    i = 0
    while (i < batch.shape[0]):
      j = 0
      pos = batch[i]
      furthest = 0
      neg = batch[i]
      for output in batch:
        if j != i and labels[j] == labels[i]:
          dist = self.getDistance(batch[j].unsqueeze(0), batch[i].unsqueeze(0))
          if dist > furthest:
            pos = batch[j]
            furthest = dist
      j = 0
      closest = furthest
      for output in batch:
        if labels[j] != labels[i]:
          dist = self.getDistance(batch[j].unsqueeze(0), batch[i].unsqueeze(0))
          if dist < closest:
            neg = batch[j]
            closest = dist

      anchor = batch[i].unsqueeze(0)
      pos = pos.unsqueeze(0)
      neg = neg.unsqueeze(0)
      running_loss += self.eval(anchor, pos, neg)
      i += 1
    
    # return average loss for batch
    return running_loss / (batch.shape[0] / 3)
    
  def getDistance(self, tensor1, tensor2):
    return torch.norm(tensor1 - tensor2, dim=1, p=None)