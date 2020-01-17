from torchvision import models
import torch
from models.model import Model

class Alexnet(Model):
    def __init__(self, loaders, loss_fn, epochs=20, pretraining=True, step_size=7, feature_extracting=False, lr=0.01):
        alexnet = models.alexnet(pretrained=pretraining)
        self.name = "Alexnet"

        super().__init__(loaders, alexnet, loss_fn, epochs, pretraining, step_size, feature_extracting, lr)

    def get_optimizer(self, lr):
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Linear(num_ftrs, 256)
        return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)