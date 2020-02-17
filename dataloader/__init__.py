from .loader import Loader
from .offline_loader import OfflineLoader
from .online_loader import OnlineLoader

__factory = {
    'triplet': OfflineLoader,
    'batchhard': OnlineLoader,
    'batchall': OnlineLoader,
    'badbatchall': OnlineLoader
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Loss has no specified dataloader:", name)
    return __factory[name](*args, **kwargs)