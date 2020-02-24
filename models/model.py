from torch.optim import lr_scheduler
import torch
import time
import logging

from models.train import train_model
from models.save_features import SaveFeatures

class Model():
  def __init__(self, dataloaders, model, loss_fn, acc_fn, epochs=20, pretrained=0, step_size=7, feature_extracting=False, lr=0.01, output_layers=256, name="model"):
    self.epochs = epochs

    self.loss_fn = loss_fn
    self.acc_fn = acc_fn
    self.loaders = dataloaders
    self.model = model
    self.name = name

    self.output_layers = output_layers

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

    self.optimizer = self.get_optimizer(lr)
    self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.1)

    # for m in self.model.modules():
    #   self.init_params(m)

  def train(self):
    start_time = time.time()
    train_model(self.loaders, self.model, self.loss_fn, self.acc_fn, self.optimizer, self.scheduler, self.epochs, name=self.name)
    logging.info('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

  def get_optimizer(self, lr):
    return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
  
  def init_params(self, m):
    if type(m)==torch.nn.Linear or type(m)==torch.nn.Conv2d:
      m.weight.data=torch.randn(m.weight.size())*.01#Random weight initialisation
