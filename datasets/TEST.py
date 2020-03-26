from datasets.data import Data

class TEST(Data):
    def __init__(self):
        TEST_dir = '/Users/greg/Data/TEST/'
        TEST_img_size = 256
        self.name = "TEST"
        super().__init__(TEST_dir, TEST_img_size)