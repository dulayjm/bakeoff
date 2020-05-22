# from os import listdir
# import pandas as pd
# from torchvision import transforms

# from datasets.dataset import TargetDataset

# class _Data():
#     def __init__(self, data_dir, image_size):
#         self.data_dir = data_dir
#         self.image_size = image_size
#         self.classes = self.sort_classes()
#         self.train_data, self.valid_data = self.get_data()

#         self.transform = transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#         self.train = Subset(self.train_data, self.data_dir, self.transform)
#         self.valid = Subset(self.valid_data, self.data_dir, self.transform)

#     def sort_classes(self):
#         classes = listdir(self.data_dir)
#         return sorted(classes, key=lambda item: (int(item.partition('.')[0])
#                                if item[0].isdigit() else float('inf'), item))

#     def get_data(self):
#         train = []
#         for class_index, label in enumerate(self.classes):
#             path = self.data_dir# + label + '/'
#             for file in listdir(path):
#                 train.append(['{}/{}'.format(label, file), label, class_index])

#         df = pd.DataFrame(train, columns=['file', 'category', 'category_id',])
#         train_data = df.sample(frac=0.7).reset_index().drop(columns=['index'])
#         valid_data = df[~df['file'].isin(train_data['file'])].reset_index().drop(columns=['index'])
#         # shuffle valid_data
#         valid_data = valid_data.sample(frac=1)
#         return train_data, valid_data
        
    
# class Subset():
#     def __init__(self, table, data_dir, transform):
#         self.table = table
#         self.data_dir = data_dir
#         self.transform = transform
#         self.set = TargetDataset(self.table, self.data_dir, transform = self.transform)

#     def shuffle(self):
#         self.table = self.table.sample(frac=1)
#         self.set = TargetDataset(self.table, self.data_dir, transform = self.transform)
