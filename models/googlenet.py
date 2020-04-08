from torchvision import models
import torch
from models.model import Model
from models.save_features import SaveFeatures

class Googlenet(Model):
    def __init__(self, loaders, loss_fn, acc_fn, epochs=20, pretrained=0, step_size=7, feature_extracting=False, lr=0.01, output_size=256, name="Googlenet", visualize="none"):
        googlenet = models.googlenet(pretrained=True)

        super().__init__(loaders, googlenet, loss_fn, acc_fn, epochs, pretrained, step_size, feature_extracting, lr, output_size, name=name, visualize=visualize)
        self.activated_features = SaveFeatures(list(self.model.modules())[-4])

    def get_optimizer(self, lr):
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, self.output_size)
        return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)