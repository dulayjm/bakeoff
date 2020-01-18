import logging
from os import system

class Loader():
  def __init__(self, data_table, dataset, batch_size):
    self.batch_size = batch_size
    self.dataset = dataset
    self.batched_data = self.makeBatches(data_table, dataset, batch_size)

  def makeBatches(self, data_table, dataset, batch_size):
    batches = []
    
    index = 0

    while index < len(data_table):
      batch = []
      # batch size is lesser between preset batch size and images remaining in dataset
      for i in range(min(batch_size, len(data_table)-index)):
        batch.append(self.getBatch(index, data_table, dataset))
        index += 1

        # log progress
        system('clear')
        print("Loading dataloader...")
        perc = index*50//len(data_table)
        print(">"*perc + "-"*(50-perc))

      batches.append(batch)
    return batches

  def getBatch(self, index, data_table, dataset):
    return dataset[index]
