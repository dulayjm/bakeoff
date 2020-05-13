from datasets.data import Data
from scipy.io import loadmat
import numpy as np
import pandas as pd

class CARS(Data):
    def __init__(self):
        CARS_dir = '/lab/vislab/DATA/CARS/'
        CARS_img_size = 256
        self.name = "CARS"
        super().__init__(CARS_dir, CARS_img_size)

    def sort_classes(self):
        classes = np.transpose(loadmat(self.data_dir + 'labels.mat')['class_names'])
        return [label[0][0] for label in classes]

    def get_data(self):
        data = np.transpose(loadmat(self.data_dir + 'labels.mat')['annotations'])
        train = []
        for img in data:
            file, x1, y1, x2, y2, class_id, test = img[0]
            class_id = class_id[0][0]
            train.append([file[0], self.classes[class_id-1], class_id])

        df = pd.DataFrame(train, columns=['file', 'category', 'category_id',])
        train_data = df.sample(frac=0.7)
        valid_data = df[~df['file'].isin(train_data['file'])]
        return train_data, valid_data