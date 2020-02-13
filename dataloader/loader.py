import logging
from os import system

class Loader():
  def __init__(self, data_table, dataset, batch_size, name="Loader"):
    self.batch_size = batch_size
    self.dataset = dataset
    self.data_table = data_table
    self.name = name

  def makeBatches(self, batch_size):
    batches = []
    
    index = 0
    batch_id = 0
    num_batches = len(self.data_table) // batch_size + 1

    while batch_id < num_batches:
      batch = []
      batch_labels = []
      # batch size is lesser between preset batch size and images remaining in dataset
      for i in range(batch_size):
        images, labels = self.getSet(index % len(self.data_table), self.data_table, self.dataset)
        batch.extend(images)
        batch_labels.extend(labels)

        index += 1

        # log progress
        system('clear')
        print("Loading dataloader...")
        perc = index*50//len(self.data_table)
        print(">"*perc + "-"*(50-perc))

      batch_id += 1
      batches.append([batch, batch_labels])
    return batches

  def getSet(self, index, data_table, dataset):
    return [dataset[index][0]], [dataset[index][1]]

  def __str__(self):
    return "{} with batch size {}".format(self.name, self.batch_size)
