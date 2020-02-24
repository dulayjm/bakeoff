import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

import logging


def euclidean_dist(inputs_):
    # Compute pairwise distance, replace by the official when merged
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class BatchAllLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(BatchAllLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist_mat = euclidean_dist(inputs)
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) ^ pos_mask


        pos_mask = pos_mask ^ eyes_.eq(1)
        pos_dist = torch.masked_select(dist_mat, pos_mask)
        neg_dist = torch.masked_select(dist_mat, neg_mask)

        num_instances = len(pos_dist)//n + 1
        num_neg_instances = n - num_instances
        pos_dist = pos_dist.reshape(len(pos_dist)//(num_instances-1), num_instances-1)
        neg_dist = neg_dist.reshape(len(neg_dist)//(num_neg_instances), num_neg_instances)

        loss = list()
        for i, pos_pair in enumerate(pos_dist):
            neg_dist_ = neg_dist[i].repeat(num_instances - 1, 1)
            pos_dist_ = pos_pair.repeat(num_neg_instances, 1)
            pos_dist_ = pos_dist_.t()
            pos_dist_ = pos_dist_.reshape(num_neg_instances * (num_instances - 1))
            neg_dist_ = neg_dist_.reshape(num_neg_instances * (num_instances - 1))

            y = neg_dist_.data.new()
            y.resize_as_(neg_dist_.data)
            y.fill_(1)
            y = Variable(y)
            loss.append(self.ranking_loss(neg_dist_, pos_dist_, y))
        loss = torch.mean(torch.stack([loss_ for loss_ in loss]))
        return loss
    
    def __str__(self):
        return "Batch All, margin = {}".format(self.margin)

def main():
    data_size = 150
    input_dim = 28
    output_dim = 256
    num_class = 15
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    print('training data is ', x.shape)
    print('initial parameters are ', w.shape)
    for i, img in enumerate(x):
        x[i] = x[i].mm(w)
    print('extracted feature is :', inputs.shape)

    # y_ = np.random.randint(num_class, size=data_size)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))
    print(BatchAllLoss(margin=0.2)(inputs, targets))


if __name__ == '__main__':
    main()