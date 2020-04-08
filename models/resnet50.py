from torchvision import models
import torchvision
import torch
import logging
from models.model import Model
from models.save_features import SaveFeatures
import sys

class Resnet50(Model):
    def __init__(self, loaders, loss_fn, acc_fn, epochs=20, pretrained=0, step_size=7, feature_extracting=False, lr=0.01, output_layers=256, name="Resnet18", visualize="none"):
        resnet = models.resnet50(pretrained=True)

        super().__init__(loaders, resnet, loss_fn, acc_fn, epochs, pretrained, step_size, feature_extracting, lr, output_layers, name=name, visualize=visualize)
        self.activated_features = SaveFeatures(self.model._modules.get('layer4'))


    def get_optimizer(self, lr):
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, self.output_layers)
        return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    
    # block can be entire model of block of layers within model
    def randomizeLastLayers(self, block, num_pretrain, layer_idx=0):
        # randomize layers once number of requested pretrained layers reached
        for layer in block.children():
            # if the layer contains layers within itself, iterate over those layers with recursion
            if (type(layer)==torch.nn.Sequential or type(layer)==torchvision.models.resnet.Bottleneck):
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