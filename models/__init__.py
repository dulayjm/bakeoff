from .alexnet import Alexnet
from .googlenet import Googlenet
from .resnet18 import Resnet18

__factory = {
    'resnet18': Resnet18,
    'googlenet': Googlenet,
    'alexnet': Alexnet,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)