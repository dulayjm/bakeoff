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
from dataloaders.offline_loader import OfflineLoader
from dataloaders.online_loader import OnlineLoader
from dataloaders.loader import Loader
from accuracy.knearest import KNN
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-loss', default='triplet', required=True,
                    help='path to dataset')
args = parser.parse_args()

data = TEST()

model_param = {
  "loaders": {},
  "loss_fn": loss.create(args.loss),
  "acc_fn": KNN(),
  "epochs": 20,
  "pretraining": False,
  "step_size": 7,
  "feature_extracting": False,
  "learning_rate": 0.001,
  "output_layers": 256,
  "name": "TEST_batch_all_20epoch_randinit"
}

# setup logging and turn off PIL plugin logging
logging.basicConfig(filename="{}.log".format(model_param["name"]), level=logging.DEBUG, format='%(asctime)s:%(name)s:%(levelname)s::  %(message)s')
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

logging.info("-"*50)
logging.info("New Model")

train_loader = OnlineLoader(data.train_data, data.train_set, 45)
valid_loader = Loader(data.valid_data, data.valid_set, batch_size=90)
model_param['loaders'] = {'train':train_loader, 'valid':valid_loader}

for param in model_param:
  logging.info("{}: {}".format(param, str(model_param[param])))

resnet = Resnet(model_param["loaders"], model_param["loss_fn"], model_param["acc_fn"], model_param["epochs"], model_param["pretraining"], 
                model_param["step_size"], model_param["feature_extracting"], model_param["learning_rate"], model_param["output_layers"], model_param["name"])
resnet.train()
