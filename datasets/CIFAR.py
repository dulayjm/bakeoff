from datasets.data import Data

class DIFAR(Data):
    def __init__(self):
        CIFAR_dir = '/Users/greg/Data/CIFAR/images/'
        CIFAR_img_size = 256
        self.name = "CIFAR"
        super().__init__(CIFAR_dir, CIFAR_img_size)
