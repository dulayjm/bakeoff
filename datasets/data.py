from scipy.io import loadmat
from PIL import Image
import requests
from io import BytesIO
from os import listdir
import numpy as np
import pandas as pd
import csv
from torchvision import transforms

from datasets.dataset import TargetDataset


class Data:
    def __init__(self):
        self.data_dir = '/lab/vislab/DATA/Hotels-50K/bakeoff/dataset'
        self.img_size = 256
        self.name = "Hotels"
        self.classes = self.sort_classes()
        self.train_data, self.valid_data = self.get_data()

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def sort_classes(self):
        """Function to retrieve only classifiers"""
        with open('dataset/train_set.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            classes = np.array(1)
            for row in reader:
                class_ = row[1]
                # append unique to classes
                if class_ not in classes: 
                    np.append(classes, class_)

        #  classes = listdir(self.data_dir)
        # return sorted(classes, key=lambda item: (int(item.partition('.')[0])
        #                        if item[0].isdigit() else float('inf'), item))
        return classes
        
    def get_data(self):
        """ Retrieve data information from the training set file """
        # data = np.transpose(loadmat(self.data_dir + 'labels.mat')['annotations'])
        # transpose permutates the dimenions of an array
        with open('dataset/train_set.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            train = np.empty(0, 50000)
            # result_array = np.empty(data_array.size) # the default dtype is float, so set dtype if it isn't float
            # for idx, line in enumerate(data_array):
            # result_array[idx] = do_stuff(line)
            for row in reader: 
                class_ = row[1]
                img_url = row[2]
                label = row[3] 
                img = None
                # access the specific file as a request
                try: 
                    response = requests.get(img_url)
                    img = Image.open(BytesIO(response.content))
                except: 
                    print("could not read url", img_url)
                if img is not None:
                    np.append(train, [class_, img_url, img, label], axis=0)
                #  np array: 
                #  _row_ class image       url                 label
                #    0   22    hotel.png   http://example.png  traffickam
                #    1   87    hotel.png   http://example.png  traffickam
                #    n   j     hotel.png   http://example.png  traffickam

        df = pd.DataFrame(train, columns=['class', 'img_url', 'image', 'label'])
        train_data = df.sample(frac=0.7)
        valid_data = df[~df['img_url'].isin(train_data['img_url'])]
        return train_data, valid_data   

# class Subset():
#     def __init__(self, table, data_dir, transform):
#         self.table = table
#         self.data_dir = data_dir
#         self.transform = transform
#         self.set = TargetDataset(self.table, self.data_dir, transform = self.transform)

#     def shuffle(self):
#         self.table = self.table.sample(frac=1)
#         self.set = TargetDataset(self.table, self.data_dir, transform = self.transform)
