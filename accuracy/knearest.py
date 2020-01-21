import torch

class KNN():
  def correct(self, test, data, model, device, top=1):
    # separate weights from labels
    weights = [item[0] for item in data]
    for w in weights:
      assert test[0] is not w, "Test image should not be in set of images to get distance from"
    weights = torch.stack(weights)
    labels = [item[1] for item in data]

    # get X closest outputs to test image
    dist = torch.norm(weights - test[0], dim=1, p=None)
    dist = torch.norm(dist, dim=1, p=None)
    dist = torch.norm(dist, dim=1, p=None)
    knn1 = dist.topk(top, largest=False)
    knn5 = dist.topk(5, largest=False)

    # return if actual test image label is contained in the top X closest image labels
    knn_labels1 = [labels[index] for index in knn1.indices.tolist()]
    knn_labels5 = [labels[index] for index in knn5.indices.tolist()]
    print(test)
    print("KNN Top 5: {}".format(knn_labels5))
    if (test[1] in knn_labels1 and test[1] in knn_labels5):
      return [1, 1]
    elif (test[1] in knn_labels5):
      return [0, 1]
    else:
      return [0, 0]
