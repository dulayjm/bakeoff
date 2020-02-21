from torch.utils.data import Dataset
from os.path import join
from PIL import Image

class TargetDataset(Dataset):
    def __init__(self, labels, root_dir, subset=False, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        fullname = join(self.root_dir, img_name)
        image = Image.open(fullname).convert('RGB')
        labels = self.labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        return image, int(labels)

    def makeTable(self, table):
        for file in listdir(path):
            train.append(['{}/{}'.format(label, file), label, class_index])

        df = pd.DataFrame(train, columns=['file', 'category', 'category_id',])
