import logging
from os import system

class Loader():
  def __init__(self, data, batch_size, name="Loader"):
    self.batch_size = batch_size
    self.data = data
    self.name = name

  def makeBatches(self, batch_size):
    batches = []
    
    index = 0
    batch_id = 0
    num_batches = len(self.data.table) // batch_size + 1

    while batch_id < num_batches:
      batch = []
      batch_labels = []
      batch_files = []
      # batch size is lesser between preset batch size and images remaining in data.set
      for i in range(batch_size):
        images, labels, fileNames = self.getSet(index % len(self.data.table))
        batch.extend(images)
        batch_labels.extend(labels)
        batch_files.extend(fileNames)

        index += 1

        # log progress
        system('clear')
        print("Loading dataloader...")
        perc = index*50//len(self.data.table)
        print(">"*perc + "-"*(50-perc))

      batch_id += 1
      batches.append([batch, batch_labels, batch_files])
    return batches

  def getSet(self, index):
    img, label, fileName = self.data.set[index]
    return [img], [label], [fileName]

  def __str__(self):
    return "{} with batch size {}".format(self.name, self.batch_size)
