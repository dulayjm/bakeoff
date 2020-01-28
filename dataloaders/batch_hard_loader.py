from random import randint
import numpy as np
import logging
from os import system

from dataloaders.loader import Loader

class BatchHardLoader(Loader):
  def __init__(self, data_table, dataset, batch_size):
    self.num_classes = len(set(data_table['category_id']))
    assert batch_size % self.num_classes == 0, "For batch hard loss, batch size must be a multiple of the number of classes"
    assert batch_size % self.num_classes == 0, "For batch hard loss, batch size must be a multiple of 3"

    super().__init__(data_table, dataset, batch_size, "Batch Hard Loader")

  def makeBatches(self, batch_size):
    batches = []
    # map each label to the index of all that label's images after shuffling
    self.data_table = self.data_table.sample(frac=1)
    map_label_indices = {label: np.flatnonzero(self.data_table['category_id'] == label).tolist() for label in self.data_table['category_id']}
    classes = list(set(self.data_table['category_id']))

    for i in range(len(self.data_table) // batch_size):
      # log progress
      system('clear')
      print("Loading dataloader...")
      perc = i*50//(len(self.data_table) // batch_size)
      print(">"*perc + "-"*(50-perc))
      
      batch = [[],[]]
      class_idx = 0
      remaining_labels = self.num_classes
      while (len(batch[0]) < batch_size):
        num_class_samples = min((batch_size - len(batch[0])) // remaining_labels, len(map_label_indices[classes[class_idx]]))
        if (num_class_samples > 1):
          batch[0].extend(self.dataset[map_label_indices[classes[class_idx]].pop(0)][0] for j in range(num_class_samples))
          batch[1].extend([classes[class_idx] for g in range(num_class_samples)])
        remaining_labels = max(remaining_labels-1, 1)
        class_idx = class_idx+1 if class_idx < len(classes)-1 else 0
      batches.append(batch)
    return batches

