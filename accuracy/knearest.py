import torch

class KNN():
  def get_acc(self, outputs, labels):
    j = 0
    corrects = 0
    for anchor in outputs:
      test_data, test_labels = self.removeAtIndex([outputs.tolist(), labels], j)
      test_data = [torch.FloatTensor(embedding) for embedding in test_data]
      anchor_label = labels[j]
      correct, _ = self.correct(anchor, anchor_label, torch.stack(test_data), test_labels)
      corrects += correct
      j += 1
    return corrects / j

  def correct(self, anchor, anchor_label, weights, labels, top=5):
    # error checking
    for w in weights:
      assert anchor is not w, "Test image should not be in set of images to get distance from"

    # get X closest outputs to test image
    dist = torch.norm(weights - anchor, dim=1, p=None)
    knn = dist.topk(top, largest=False)

    # return if actual test image label is contained in the top X closest image labels
    knn_labels = [labels[index] for index in knn.indices.tolist()]
    # top match is the correct label
    if (anchor_label == knn_labels[0]):
      return [1, 1]
    # correct label is in top 5
    elif (anchor_label in knn_labels):
      return [0, 1]
    # no match
    else:
      return [0, 0]

  def removeAtIndex(self, lists, j):
    removed = []
    for each in lists:
      removed.append(each[:j] + each[j+1:])
    return removed

