import codecs
import torch
import torch.utils.data as data
from PIL import Image


def to_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def to_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def labels(path):
    with open(path, 'rb') as f:
        d = f.read()
        return torch.LongTensor([to_byte(b) for b in d[8:]])


def images(path):
    with open(path, 'rb') as f:
        d = f.read()
        rows = to_int(d[8:12])
        cols = to_int(d[12:16])
        imgs = []
        idx = 16
        for l in range(to_int(d[4:8])):
            img = []
            imgs.append(img)
            for r in range(rows):
                row = []
                img.append(row)
                for c in range(cols):
                    row.append(to_byte(d[idx]))
                    idx += 1
        return torch.ByteTensor(imgs).view(-1, 28, 28)


class MNIST(data.Dataset):
    def __init__(
        self,
        path_to_raw,
        path_to_processed,
        train=True,
        transform=None,
        target_transform=None
    ):
        training = (
            images(path_to_raw / 'train-images-idx3-ubyte'),
            labels(path_to_raw / 'train-labels-idx1-ubyte'))
        test = (
            images(path_to_raw / 't10k-images-idx3-ubyte'),
            labels(path_to_raw / 't10k-labels-idx1-ubyte'))

        with open(path_to_processed / 'training.pt', 'wb') as f:
            torch.save(training, f)
        with open(path_to_processed / 'test.pt', 'wb') as f:
            torch.save(test, f)

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.images, self.labels = torch.load(path_to_processed / 'training.pt')
        else:
            self.images, self.labels = torch.load(path_to_processed / 'test.pt')

    def __getitem__(self, index):
        image, target = self.images[index], self.labels[index]
        image = Image.fromarray(image.numpy(), mode='L')

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)
