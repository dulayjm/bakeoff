from .triplet import TripletLoss
from .BatchHard import BatchHardLoss
from .BatchAll import BatchAllLoss
from .BadBatchAll import BadBatchAllLoss

__factory = {
    'triplet': TripletLoss,
    'batchhard': BatchHardLoss,
    'batchall': BatchAllLoss,
    'badbatchall': BadBatchAllLoss
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)