from datasets.data import Data

class DIFAR(Data):
    def __init__(self):
        CIFAR_dir = '/student/rolwesg/Data/CIFAR10/'
        CIFAR_img_size = 32
        self.name = "CIFAR"
        super().__init__(CIFAR_dir, CIFAR_img_size)
