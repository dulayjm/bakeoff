import numpy as np
import torch

from loss.loss import Loss

class TripletLoss(Loss):
   def __init__(self, margin=1.0):
      super().__init__(torch.nn.TripletMarginLoss(margin=margin, p=2))
      self.name = "Triplet Loss"

   def getLoss(self, batch, model, device="cpu"):
      running_loss = 0

      for triplet in batch:
         assert isinstance(triplet, list) and len(triplet) is 3, "Triplet loss requires batches to be composed of triplets"
         assert triplet[0][1] == triplet[1][1], "Anchor and positive must be of the same label"
         assert triplet[0][1] != triplet[2][1], "Anchor and negative must be of different labels"
         
         # send to GPU if available and increase dimensions to fit model
         anchor = triplet[0][0].to(device).unsqueeze(0)
         pos = triplet[1][0].to(device).unsqueeze(0)
         neg = triplet[2][0].to(device).unsqueeze(0)

         running_loss += self.eval(model(anchor), model(pos), model(neg))
      
      # return average loss for batch
      return running_loss / len(batch)
