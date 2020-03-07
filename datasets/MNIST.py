from datasets.data import Data

class MNIST(Data):
    def __init__(self):
        MNIST_dir = '/Users/greg/Data/MNIST/training/'
        MNIST_img_size = 28
        self.name = "MNIST"
        super().__init__(MNIST_dir, MNIST_img_size)
