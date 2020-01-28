import logging

from models.resnet import Resnet
from models.alexnet import Alexnet
from models.googlenet import Googlenet
from datasets.CUB import CUB
from datasets.CARS import CARS
from datasets.TEST import TEST
from datasets.MNIST import MNIST
from loss.triplet import TripletLoss
from loss.batch_hard import BatchHardLoss
from dataloaders.triplet_loader import TripletLoader
from dataloaders.batch_hard_loader import BatchHardLoader
from dataloaders.loader import Loader
from accuracy.knearest import KNN
import numpy as np

data = MNIST()

model_param = {
  "loaders": {},
  "loss_fn": TripletLoss(margin=0.5),
  "acc_fn": KNN(),
  "epochs": 50,
  "pretraining": True,
  "step_size": 7,
  "feature_extracting": False,
  "learning_rate": 0.001,
  "output_layers": 256,
  "name": "testing_batch_hard_loader_MNIST"
}

logging.basicConfig(filename="{}.log".format(model_param["name"]), level=logging.DEBUG, format='%(asctime)s:%(levelname)s::  %(message)s')

train_loader = BatchHardLoader(data.train_data, data.train_set, 50)
valid_loader = Loader(data.valid_data, data.valid_set, batch_size=50)
model_param['loaders'] = {'train':train_loader, 'valid':valid_loader}
print(str(model_param['loss_fn']))
logging.info("New model: {}".format(model_param))
logging.info("Train Batch Size: {}".format(model_param['loaders']['train'].batch_size))

resnet = Resnet(model_param["loaders"], model_param["loss_fn"], model_param["acc_fn"], model_param["epochs"], model_param["pretraining"], 
                model_param["step_size"], model_param["feature_extracting"], model_param["learning_rate"], model_param["output_layers"], model_param["name"])
resnet.train()
