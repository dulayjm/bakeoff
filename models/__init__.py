import dataset as Hotels

# __factory = {
#     'Hotels': Hotels,
# }


# def names():
#     return sorted(__factory.keys())


def create(name, *args, **kwargs):
    # if name not in __factory:
    #     raise KeyError("Unknown model:", name)
    return Hotels(*args, **kwargs)