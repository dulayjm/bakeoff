import torch
import sys

class KNN():
  def get_acc(self, outputs, labels):
    j = 0
    corrects = 0
    pairs = []
    for anchor in outputs:
      anchor_label = labels[j]
      # correct requires a list of tensors
      if torch.cuda.is_available():
          outputs = [torch.cuda.FloatTensor(output) for output in outputs]
      else:
          outputs = [torch.FloatTensor(output) for output in outputs]
      correct, top_idx = self.correct(anchor, anchor_label, torch.stack(outputs), labels)
      # appends the anchor index and most similar image index to be used for visualizations
      pairs.append([j, top_idx, correct])
      corrects += correct
      j += 1
    return corrects / j, pairs

  def correct(self, anchor, anchor_label, weights, labels):
    # error checking
    for w in weights:
      assert anchor is not w, "Test image should not be in set of images to get distance from"

    # get closest outputs to test image
    dist = torch.norm(weights - anchor, dim=1, p=None)
    knn = dist.topk(5, largest=False)

    # get indices of most similar images, excluding the first one as that will always be the image itself
    knn_indices = knn.indices.tolist()[1:]
    # get labels of most similar images
    knn_labels = [labels[index] for index in knn_indices]
    if (anchor_label == knn_labels[0]):
      return 1, knn_indices[0]
    else:
      return 0, knn_indices[0]

  def __str__(self):
    return "K Nearest Neighbor"
