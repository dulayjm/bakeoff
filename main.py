from __future__ import absolute_import, print_function
import numpy as np
import argparse
import logging
import sys

import models
import datasets
import loss
import dataloader
import accuracy

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-model', default='resnet', required=False,
                    help='model architecture')
parser.add_argument('-loss_fn', default='batchall', required=False,
                    help='loss function')
parser.add_argument('-acc_fn', default='knn', required=False,
                    help='accuracy evaluation function')
parser.add_argument('-epochs', default=50, required=False,
                    help='number of epochs')
parser.add_argument('-pretrain', default=False, required=False,
                    help='pretrain on ImageNet?')
parser.add_argument('-step_size', default=7, required=False,
                    help='number of epochs between decreasing learning rate')
parser.add_argument('-feature_extracting', default=False, required=False,
                    help='should the model be feature extracting')
parser.add_argument('-lr', default=0.001, required=False,
                    help='optimizer learning rate')
parser.add_argument('-output_layers', default=256, required=False,
                    help='number of output layers')
parser.add_argument('-name', default='model', required=True,
                    help='custom model name')
parser.add_argument('-dataset', default='MNIST', required=True,
                    help='samples per training batch')
parser.add_argument('-batch_size', default=45, required=True,
                    help='samples per training batch')
args = parser.parse_args()

data = datasets.create(args.dataset)

train_loader = dataloader.create(args.loss_fn, data.train_data, data.train_set, int(args.batch_size))
valid_loader = dataloader.create(args.acc_fn, data.valid_data, data.valid_set, int(args.batch_size))

model_param = {
  "dataset": args.dataset,
  "batch size": int(args.batch_size),
  "loaders": {'train':train_loader, 'valid':valid_loader},
  "loss_fn": loss.create(args.loss_fn),
  "acc_fn": accuracy.create(args.acc_fn),
  "epochs": int(args.epochs),
  "pretraining": bool(args.pretrain),
  "step_size": int(args.step_size),
  "feature_extracting": bool(args.feature_extracting),
  "learning_rate": float(args.lr),
  "output_layers": int(args.output_layers),
  "name": args.name
}

model = models.create(args.model,
                model_param["loaders"], 
                model_param["loss_fn"], 
                model_param["acc_fn"], 
                model_param["epochs"], 
                model_param["pretraining"], 
                model_param["step_size"], 
                model_param["feature_extracting"], 
                model_param["learning_rate"], 
                model_param["output_layers"], 
                model_param["name"]
              )

# setup logging and turn off PIL plugin logging
logging.basicConfig(filename="logs/{}.log".format(model_param["name"]), level=logging.DEBUG, format='%(asctime)s:%(name)s:%(levelname)s::  %(message)s')
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

logging.info("-"*50)
logging.info("New Model")

for param in model_param:
  logging.info("{}: {}".format(param, str(model_param[param])))

model.train()
