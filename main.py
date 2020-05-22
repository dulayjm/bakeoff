from __future__ import absolute_import, print_function
# from ray import tune
import numpy as np
import argparse
import logging
import sys
import os

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
parser.add_argument('-pretrain', default=0, required=False,
                    help='number of pretrained layers')
parser.add_argument('-step_size', default=7, required=False,
                    help='number of epochs between decreasing learning rate')
parser.add_argument('-feature_extracting', default=False, required=False,
                    help='should the model be feature extracting')
parser.add_argument('-lr', default=0.01, required=False,
                    help='optimizer learning rate')
parser.add_argument('-output_size', default=256, required=False,
                    help='output layer size')
parser.add_argument('-name', default='model', required=True,
                    help='custom model name')
parser.add_argument('-dataset', default='MNIST', required=True,
                    help='samples per training batch')
parser.add_argument('-batch_size', default=45, required=True,
                    help='samples per training batch')
parser.add_argument('-visualize', default='none', required=False,
                    help='class to visualize')
args = parser.parse_args()

data = datasets.create(args.dataset)

train_loader = dataloader.create(args.loss_fn, data.train, int(args.batch_size))
valid_loader = dataloader.create(args.loss_fn, data.valid, int(args.batch_size))

def train_model(config):
    model = models.create(args.model,
                config["loaders"], 
                config["loss_fn"], 
                config["acc_fn"], 
                config["epochs"], 
                config["pretrained"], 
                config["step_size"], 
                config["feature_extracting"], 
                config["learning_rate"], 
                config["output_size"], 
                config["name"],
                # config["visualization class"]
              )

    if not os.path.exists("results/{}/".format(config["name"])):
        os.makedirs("results/{}/".format(config["name"]))

    # setup logging and turn off PIL plugin logging
    logging.basicConfig(filename="results/{}/training.log".format(config["name"]), level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s::  %(message)s')
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    logging.info("-"*50)
    logging.info("New Model")

    for param in config:
      logging.info("{}: {}".format(param, str(config[param])))
    model.train()
#     tune.track.log(mean_accuracy=model.train())

config = {
  "dataset": args.dataset,
  "batch size": int(args.batch_size),
  "loaders": {'train':train_loader, 'valid':valid_loader},
  "loss_fn": loss.create(args.loss_fn),
  "acc_fn": accuracy.create(args.acc_fn),
  "epochs": int(args.epochs),
  "pretrained": int(args.pretrain),
  "step_size": int(args.step_size),
  "feature_extracting": bool(args.feature_extracting),
  "learning_rate": float(args.lr),
#     tune.grid_search([0.001, 0.01, 0.1]),
  "output_size": int(args.output_size),
  "name": args.name,
  "visualization class": args.visualize
}

for item in config: 
  print("This is the item in config: ", config[item])


train_model(config)
# analysis = tune.run(train_model, 
#                     config=config, 
#                     resources_per_trial={
#                          "cpu": 1,
#                          "gpu": 1
#                      }
#                    )
# print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))