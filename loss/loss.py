import numpy as np
import torch

class Loss():
   def __init__(self, loss_fn=torch.nn.CrossEntropyLoss(), name="Softmax Cross Entropy"):
      self.eval = loss_fn
      self.name = name

   def getLoss(self, outputs, labels, device="cpu"):
      return self.eval(outputs.to(device), labels.to(device))

   def __str__(self):
      return self.name
