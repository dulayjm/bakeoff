from torchvision import models
import torch
from models.model import Model
from models.save_features import SaveFeatures

class Alexnet(Model):
    def __init__(self, loaders, loss_fn, acc_fn, epochs=20, pretrained=0, step_size=7, feature_extracting=False, lr=0.01, output_layers=256, name="Alexnet", visualize="none"):
        alexnet = models.alexnet(pretrained=True)

        super().__init__(loaders, alexnet, loss_fn, acc_fn, epochs, pretrained, step_size, feature_extracting, lr, output_layers, name=name, visualize=visualize)
        self.activated_features = SaveFeatures(dict(list(self.model.modules())[0].features.named_children())['10'])

    def get_optimizer(self, lr):
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Linear(num_ftrs, self.output_layers)
        return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)