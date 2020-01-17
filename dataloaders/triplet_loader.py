from random import randint
import numpy as np
from os import system

from dataloaders.loader import Loader

class TripletLoader(Loader):
  def __init__(self, data_table, dataset, batch_size):
    super().__init__(data_table, dataset, batch_size)

  def makeBatches(self, data_table, dataset, batch_size):
    batches = []
    # map each label to the index of all that label's images
    map_label_indices = {label: np.flatnonzero(data_table['category_id'] == label).tolist() for label in data_table['category_id']}
    
    index = 0
    while index < len(data_table):
      batch = []

      # batch size is lesser between preset batch size and images remaining in dataset
      for i in range(min(batch_size, len(data_table)-index)):
        system('clear')
        print("Loading data...")
        print("{}/{}".format(index, len(data_table)))

        # set anchor to next image in data table
        anchor_class_id = data_table.loc[index, 'category_id']
        neg_class_id = anchor_class_id
        #select different class for negative example
        while (anchor_class_id == neg_class_id):
          neg_class_id = np.random.choice(data_table['category_id'], 1, replace=False)[0]
        assert anchor_class_id != neg_class_id, "Negative example must be of a different class than anchor."

        # randomly select a positive and a negative image
        pos_idx = np.random.choice(map_label_indices[anchor_class_id], 1, replace=False)[0]
        neg_idx = np.random.choice(map_label_indices[neg_class_id], 1, replace=False)[0]
        batch.append([dataset[index], dataset[pos_idx], dataset[neg_idx]])
        index += 1

      batches.append(batch)
    return batches
