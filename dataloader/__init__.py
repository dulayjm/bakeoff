from .loader import Loader
from .online_loader import OnlineLoader

__factory = {
    'batchhard': OnlineLoader,
    'batchall': OnlineLoader,
    'badbatchall': OnlineLoader,
    'ephn': OnlineLoader
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Loss has no specified dataloader:", name)
    return __factory[name](*args, **kwargs)