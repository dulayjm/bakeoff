from torch.optim import lr_scheduler
import time

from models.train import train_model

class Model():
  def __init__(self, dataloaders, model, loss_fn, epochs=20, step_size=7, pretraining=True, feature_extracting=False, lr=0.01):
    self.epochs = epochs

    self.loss_fn = loss_fn
    self.loaders = dataloaders
    self.model = model

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

    self.optimizer = self.get_optimizer(lr)
    self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.1)

    self.trained = False

  def train(self):
    start_time = time.time()
    train_model(self.loaders, self.model, self.loss_fn, self.optimizer, self.scheduler, self.epochs)
    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

  def get_optimizer(self, lr):
    return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

