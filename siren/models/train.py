import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path


STATS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def new_data_loader(path, image_size = 64, batch_size = 128):
    folder = ImageFolder(
        path,
        transform=tt.Compose([
            tt.Resize(image_size),
            tt.CenterCrop(image_size),
            tt.ToTensor(),
            tt.Normalize(*STATS)]))
    return DataLoader(
        folder,
        batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True)


def denorm(img):
    return img * STATS[1][0] + STATS[0][0]
