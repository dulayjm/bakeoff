import torch
from visualizer import visualize

class KNN():
  def get_acc(self, outputs, hook, fileNames, labels, size):
    j = 0
    corrects = 0
    for anchor in outputs:
      test_data, test_labels = self.removeAtIndex([outputs.tolist(), labels.tolist()], j)
      test_data = [torch.FloatTensor(embedding) for embedding in test_data]
      anchor_label = labels[j]
      correct, top_idx = self.correct(anchor, anchor_label, torch.stack(test_data), test_labels)
      visualize(hook.features[j], fileNames[j], hook.features[top_idx], fileNames[top_idx], size)
      corrects += correct
      j += 1
    return corrects / j

  def correct(self, anchor, anchor_label, weights, labels):
    # error checking
    for w in weights:
      assert anchor is not w, "Test image should not be in set of images to get distance from"

    # get closest outputs to test image
    dist = torch.norm(weights - anchor, dim=1, p=None)
    knn = dist.topk(1, largest=False)

    # return if actual test image label is contained in the closest image labels
    knn_indices = knn.indices.tolist()
    knn_labels = [labels[index] for index in knn_indices]
    if (anchor_label in knn_labels):
      return 1, knn_indices[0]
    else:
      return 0, knn_indices[0]

  def removeAtIndex(self, lists, j):
    removed = []
    for each in lists:
      removed.append(each[:j] + each[j+1:])
    return removed

  def __str__(self):
    return "K Nearest Neighbor"
