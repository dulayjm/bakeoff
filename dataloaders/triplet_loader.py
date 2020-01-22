from random import randint
import numpy as np
from os import system

from dataloaders.loader import Loader

class TripletLoader(Loader):
  def __init__(self, data_table, dataset, batch_size):
    # map each label to the index of all that label's images
    self.map_label_indices = {label: np.flatnonzero(data_table['category_id'] == label).tolist() for label in data_table['category_id']}
    
    super().__init__(data_table, dataset, batch_size)

  def getSet(self, index, data_table, dataset):
    # set anchor to next image in data table
    anchor_class_id = data_table.loc[index, 'category_id']
    neg_class_id = anchor_class_id
    # select different class for negative example
    while (anchor_class_id == neg_class_id):
      neg_class_id = np.random.choice(data_table['category_id'], 1, replace=False)[0]
    assert anchor_class_id != neg_class_id, "Negative example must be of a different class than anchor."

    # randomly select a positive and a negative image
    pos_idx = np.random.choice(self.map_label_indices[anchor_class_id], 1, replace=False)[0]
    neg_idx = np.random.choice(self.map_label_indices[neg_class_id], 1, replace=False)[0]

    # return [images], [labels]
    return [dataset[index][0], dataset[pos_idx][0], dataset[neg_idx][0]], [dataset[index][1], dataset[pos_idx][1], dataset[neg_idx][1]]
