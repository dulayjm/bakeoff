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

        self.train = Subset(self.train_data, self.data_dir, self.transform)
        self.valid = Subset(self.valid_data, self.data_dir, self.transform)

    def sort_classes(self):
        """Function to retrieve only classifiers"""
        # with open('dataset/train_set.csv', newline='') as csvfile:
        #     reader = csv.reader(csvfile)
        #     classes = np.array(1)
        #     for row in reader:
        #         class_ = row[1]
        #         # append unique to classes
        #         if class_ not in classes: 
        #             np.append(classes, class_)


        train = np.genfromtxt ('/Users/justindulay/research/infilling/bakeoff/dataset/train_set.csv', delimiter=",")
        classes = train[:,1]   
        # make unique
        result = np.empty([1])
        for c in classes: 
            if c not in result: 
                np.append(result, c)

        #  classes = listdir(self.data_dir)
        # return sorted(classes, key=lambda item: (int(item.partition('.')[0])
        #                        if item[0].isdigit() else float('inf'), item))
        return result
        
    def get_data(self):
        """Retrieve data information from the training set file"""

        csv = np.genfromtxt('dataset/train_set.csv', delimiter=",")
        classes = csv[:,1]   
        images = csv[:,2]
        labels = csv[:,3]

        # train = np.concatenate((classes,images,labels),axis=1)
        train = np.column_stack((classes,images,labels))
        #    np array: 
        #    row  class  image      url                 label
        #    0    22     hotel.png  http://example.png  traffickam
        #    1    87     hotel.png  http://example.png  traffickam
        #    n    j      hotel.png  http://example.png  traffickam


        df = pd.DataFrame(train, columns=['classes', 'image', 'label'])
        train_data = df.sample(frac=0.7)
        valid_data = df[~df['image'].isin(train_data['image'])]
        return train_data, valid_data

class Subset():
    def __init__(self, table, data_dir, transform):
        self.table = table
        self.data_dir = data_dir
        self.transform = transform
        self.set = TargetDataset(self.table, self.data_dir, transform = self.transform)

    def shuffle(self):
        self.table = self.table.sample(frac=1)
        self.set = TargetDataset(self.table, self.data_dir, transform = self.transform)