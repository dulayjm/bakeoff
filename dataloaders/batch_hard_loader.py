from random import randint
import numpy as np
from os import system

from dataloaders.loader import Loader

class BatchHardLoader(Loader):
  def __init__(self, data_table, dataset, batch_size, num_classes):
    # map each label to the index of all that label's images
    self.map_label_indices = {label: np.flatnonzero(data_table['category_id'] == label).tolist() for label in data_table['category_id']}
    
    self.num_classes = num_classes
    assert batch_size % num_classes == 0, "For batch hard loss, batch size must be a multiple of the number of classes"

    super().__init__(data_table, dataset, batch_size)

    self.num_batches = len(self.data_table) // batch_size

  def makeBatches(self, batch_size):
    batches = []

    for j in range(self.num_batches):
      batch = []
      batch_labels = []
      
      # batch size is lesser between preset batch size and images remaining in dataset
      for label in self.map_label_indices:
        images, labels = self.getSet(label, self.data_table, self.dataset, len(batch) - self.batch_size)
        batch.extend(images)
        batch_labels.extend(labels)

      # log progress
      system('clear')
      print("Loading dataloader...")
      perc = j*50//num_batches
      print(">"*perc + "-"*(50-perc))

      batches.append([batch, batch_labels])
    return batches

  def getSet(self, label, data_table, dataset, size_remaining):
    label_indices = self.map_label_indices

    images, labels = [], []
    for i in range(min(len(label_indices[label]) // self.num_batches, size_remaining)):
      new_img = label_indices[label].pop(0)
      images.append(dataset[new_img][0])
      labels.append(label)

    # return [images], [labels]
    return images, labels