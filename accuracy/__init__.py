from .knearest import KNN

__factory = {
  'knn': KNN
}


def names():
  return sorted(__factory.keys())


def create(name, *args, **kwargs):
  if name not in __factory:
      raise KeyError("Unknown accuracy function:", name)
  return __factory[name](*args, **kwargs)