from models.resnet import Resnet
from models.alexnet import Alexnet
from datasets.CUB import CUB
from datasets.CARS import CARS
from datasets.TEST import TEST
from loss.triplet import TripletLoss
from dataloaders.triplet_loader import TripletLoader
from dataloaders.loader import Loader
import numpy as np

data = TEST()

train_loader = TripletLoader(data.train_data, data.train_set, batch_size=25)
valid_loader = TripletLoader(data.valid_data, data.valid_set, batch_size=25)
loaders = {'train':train_loader, 'valid':valid_loader}

param = {
  "loaders": loaders,
  "loss_fn": TripletLoss(margin=0.3),
  "epochs": 50,
  "pretraining": True,
  "step_size": 7,
  "feature_extracting": False,
  "learning_rate": 0.001,
}

resnet = Resnet(param["loaders"], param["loss_fn"], param["epochs"], param["pretraining"], 
                param["step_size"], param["feature_extracting"], param["learning_rate"])
resnet.train()
