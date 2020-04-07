from torch.optim import lr_scheduler
import torch
import torchvision
import time
import logging
import sys

from models.train import train_model
from models.save_features import SaveFeatures

class Model():
  def __init__(self, dataloaders, model, loss_fn, acc_fn, epochs=20, pretrained=0, step_size=7, feature_extracting=False, lr=0.01, output_layers=256, name="model", visualize="none"):
    self.epochs = epochs

    self.loss_fn = loss_fn
    self.acc_fn = acc_fn
    self.loaders = dataloaders
    self.model = model
    self.name = name
    self.classOfInterest = visualize

    self.output_layers = output_layers

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

    self.optimizer = self.get_optimizer(lr)
    self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.1)

    self.randomizeLastLayers(self.model, pretrained)
    sys.exit()

  # block can be entire model of block of layers within model
  def randomizeLastLayers(self, block, num_pretrain, layer_idx=0):
    # randomize layers once number of requested pretrained layers reached
    for layer in block.children():
      # if the layer contains layers within itself, iterate over those layers with recursion
      if (type(layer)==torch.nn.Sequential or type(layer)==torchvision.models.resnet.BasicBlock):
        layer_idx = self.randomizeLastLayers(layer, num_pretrain, layer_idx)
      else:
        # only consider linear and convolutional layers for randomization
        if type(layer)==torch.nn.Linear or type(layer)==torch.nn.Conv2d:
          if (layer_idx >= num_pretrain):
            logging.debug(layer, ' of index ', layer_idx, ' randomized')
            layer.weight.data=torch.randn(layer.weight.size())*.01 #Random weight initialisation
            layer_idx += 1
          else:
            logging.debug(layer, ' of index ', layer_idx, ' is pretrained')
            layer_idx += 1
    # return layer index so recursive calls can keep track
    print(layer_idx)
    return layer_idx

  def train(self):
    start_time = time.time()
    train_model(
      self.loaders,
      self.model,
      self.loss_fn,
      self.activated_features,
      self.acc_fn,
      self.optimizer,
      self.scheduler,
      self.epochs,
      name=self.name,
      classOfInterest=self.classOfInterest
    )
    logging.info('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

  def get_optimizer(self, lr):
    return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
  
  # returns 1 if layer was randomized, 0 if was not
  def init_params(self, m):
    if type(m)==torch.nn.Linear or type(m)==torch.nn.Conv2d:
      m.weight.data=torch.randn(m.weight.size())*.01 #Random weight initialisation
      return 1
    else:
      return 0