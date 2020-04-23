from random import randint
import numpy as np
import logging
from os import system

from .loader import Loader

class OnlineLoader(Loader):
  def __init__(self, data, batch_size):
    self.num_classes = len(set(data.table['category_id']))
    assert batch_size % 16 == 0, "For Online Loader, batch size must be a multiple of 16"

    super().__init__(data, batch_size, "Online Triplets Loader")

  def makeBatches(self, batch_size):
    self.data.shuffle()
    batches = []
    # map each label to the index of all that label's images after shuffling
    map_label_indices = {label: np.flatnonzero(self.data.table['category_id'] == label).tolist() for label in self.data.table['category_id']}
    classes = list(set(self.data.table['category_id']))

    index = 0
    class_idx = 0
    while (index < len(self.data.table)):
      batch = [[],[],[]]
      for i in range(batch_size):
        num_class_samples = min(16, batch_size - len(batch[0]))
        for g in range(num_class_samples):
          if (len(map_label_indices[classes[class_idx]]) == 0):
            map_label_indices[classes[class_idx]] = np.flatnonzero(self.data.table['category_id'] == classes[class_idx]).tolist()
          
          img, label, fileName = self.data.set[map_label_indices[classes[class_idx]].pop(0)]
          batch[0].append(img)
          batch[1].append(label)
          batch[2].append(fileName)
          index += 1

          # log progress
          system('clear')
          print("Loading dataloader...")
          perc = index*50//len(self.data.table)
          print(">"*perc + "-"*(50-perc))

        class_idx = class_idx+1 if class_idx < len(classes)-1 else 0
      batches.append(batch)
    return batches

