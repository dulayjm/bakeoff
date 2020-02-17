from .CARS import CARS
from .CUB import CUB
from .MNIST import MNIST
from .TEST import TEST

__factory = {
    'CARS': CARS,
    'CUB': CUB,
    'MNIST': MNIST,
    'TEST': TEST
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](*args, **kwargs)