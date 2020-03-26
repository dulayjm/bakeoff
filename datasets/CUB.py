from datasets.data import Data

class CUB(Data):
    def __init__(self):
        CUB_dir = '/student/rolwesg/Data/CUB/images/'
        CUB_img_size = 256
        self.name = "CUB"
        super().__init__(CUB_dir, CUB_img_size)
