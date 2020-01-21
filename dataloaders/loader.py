import logging
from os import system

class Loader():
  def __init__(self, data_table, dataset, batch_size):
    self.batch_size = batch_size
    self.dataset = dataset
    self.data_table = data_table

  def makeBatches(self, batch_size):
    batches = []
    
    index = 0

    while index < len(self.data_table):
      batch = []
      batch_labels = []
      # batch size is lesser between preset batch size and images remaining in dataset
      for i in range(min(batch_size, len(self.data_table)-index)):
        images, labels = self.getSet(index, self.data_table, self.dataset)
        batch.extend(images)
        batch_labels.extend(labels)

        index += 1

        # log progress
        system('clear')
        print("Loading dataloader...")
        perc = index*50//len(self.data_table)
        print(">"*perc + "-"*(50-perc))

      batches.append([batch, batch_labels])
    return batches

  def getSet(self, index, data_table, dataset):
    return [dataset[index][0]], [dataset[index][1]]
