from torchvision import models
import torch
from models.model import Model

class Alexnet(Model):
    def __init__(self, loaders, loss_fn, acc_fn, epochs=20, pretraining=True, step_size=7, feature_extracting=False, lr=0.01, output_layers=256, name="Alexnet"):
        alexnet = models.alexnet(pretrained=pretraining)

        super().__init__(loaders, alexnet, loss_fn, acc_fn, epochs, pretraining, step_size, feature_extracting, lr, output_layers, name)

    def get_optimizer(self, lr):
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Linear(num_ftrs, self.output_layers)
        return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)