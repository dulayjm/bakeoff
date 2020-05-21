from datasets.data import Data

class Hotels(Data):
    def __init__(self):
        hotels_dir = '/lab/vislab/DATA/Hotels-50K/bakeoff/dataset/'
        hotel_img_size = 256
        self.name = "Hotels"
        super().__init__(hotels_dir, hotel_img_size)
