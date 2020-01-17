import torch

class KNN():
  def correct(self, test, data, model, device, top=1):
    weights = [item[0] for item in data]
    for w in weights:
      assert test[0] is not w, "Test image should not be in set of images to get distance from"
    weights = torch.stack(weights)
    labels = [item[1] for item in data]

    dist = torch.norm(weights - test[0], dim=1, p=None)
    dist = torch.norm(dist, dim=1, p=None)
    dist = torch.norm(dist, dim=1, p=None)
    knn = dist.topk(top, largest=False)

    knn_labels = [labels[index] for index in knn.indices.tolist()]
    if (test[1] in knn_labels):
      return 1
    else:
      return 0
