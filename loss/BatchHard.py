import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import logging
from device import device

# compute the distance matrix of outputs
def euclidean_distances(X):
  batch_size = X.size(0)
  # XX the row norm of X, so is YY
  XX = torch.pow(torch.norm(X, dim=1).to(device), 2)
  XX = XX.repeat(batch_size, 1)

  distances = X.mm(X.t())
  distances *= -2
  distances += XX
  distances += XX.t()

  # Ensure that distances between vectors and themselves are set to 0.0.
  # This may not be the case due to floating point rounding errors.
  I_mat = torch.eye(batch_size).to(device)
  mask = I_mat.ge(0.5)
  distances = distances.masked_fill(Variable(mask), 0)

  distances = torch.clamp(distances, min=0)
  distances = torch.sqrt(distances)
  return distances

def compute_ID_mat(label):
  size_ = label.size(0)
  label_mat = label.repeat(size_, 1)
  mask_ = label_mat == label_mat.t()
  # change datatype form byte to int
  ID_mat = Variable(torch.zeros(size_, size_).to(device))
  ID_mat = ID_mat.masked_fill(mask_, 1)
  return ID_mat


class BatchHardLoss(nn.Module):
  def __init__(self, margin=1):
    super(BatchHardLoss, self).__init__()
    self.margin = margin

  def forward(self, outputs, labels):
    batch_size = outputs.size(0)
    dist_mat = euclidean_distances(outputs)
    ID_mat = compute_ID_mat(labels)

    pos_dist_mat = Variable(torch.zeros(batch_size, batch_size).to(device))
    pos_dist_mat = torch.addcmul(pos_dist_mat, 1, ID_mat, dist_mat)

    neg_dist_mat = Variable(torch.zeros(batch_size, batch_size).to(device))
    neg_dist_mat = torch.addcmul(neg_dist_mat, 1, 1-ID_mat, dist_mat)
    mask_ = (neg_dist_mat == 0)
    neg_dist_mat.masked_fill_(mask_, 10000)

    hard_pos = torch.max(pos_dist_mat, dim=0)[0]
    hard_neg = torch.min(neg_dist_mat, dim=0)[0]

    triplet_losses = torch.clamp(hard_pos - hard_neg + self.margin, min=0)
    
    return torch.sum(triplet_losses)/triplet_losses.size(0)

  def __str__(self):
    return "Batch Hard, margin = {}".format(self.margin)