import numpy as np
import torch

class Loss():
   def __init__(self, loss_fn=torch.nn.CrossEntropyLoss()):
      self.eval = loss_fn

   def getLoss(self, outputs, labels, device="cpu"):
      return self.eval(outputs.to(device), labels.to(device))
