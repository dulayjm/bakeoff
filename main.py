from __future__ import absolute_import, print_function
import argparse
import logging
import sys

from models.resnet import Resnet
from models.alexnet import Alexnet
from models.googlenet import Googlenet
from datasets.CUB import CUB
from datasets.CARS import CARS
from datasets.TEST import TEST
from datasets.MNIST import MNIST
import loss
import dataloader
import accuracy
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-loss', default='batchall', required=False,
                    help='loss function')
parser.add_argument('-acc', default='knn', required=False,
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
args = parser.parse_args()

data = TEST()

train_loader = dataloader.create(args.loss, data.train_data, data.train_set, 45)
valid_loader = dataloader.create(args.acc, data.valid_data, data.valid_set, batch_size=132)

model_param = {
  "loaders": {'train':train_loader, 'valid':valid_loader},
  "loss_fn": loss.create(args.loss),
  "acc_fn": accuracy.create(args.acc),
  "epochs": int(args.epochs),
  "pretraining": bool(args.pretrain),
  "step_size": int(args.step_size),
  "feature_extracting": bool(args.feature_extracting),
  "learning_rate": int(args.lr),
  "output_layers": int(args.output_layers),
  "name": args.name
}

model = Resnet(model_param["loaders"], model_param["loss_fn"], model_param["acc_fn"], model_param["epochs"], model_param["pretraining"], 
                model_param["step_size"], model_param["feature_extracting"], model_param["learning_rate"], model_param["output_layers"], model_param["name"])

# setup logging and turn off PIL plugin logging
logging.basicConfig(filename="{}.log".format(model_param["name"]), level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s::  %(message)s')
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

logging.info("-"*50)
logging.info("New Model")

for param in model_param:
  logging.info("{}: {}".format(param, str(model_param[param])))

model.train()
