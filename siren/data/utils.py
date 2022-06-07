import torchvision.transforms as tt
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image


def load(
    path,
    image_size = 64,
    batch_size = 128
):
    transform = tt.Compose([
        tt.Resize(image_size),
        tt.CenterCrop(image_size),
        tt.ToTensor(),
        tt.Normalize(*STATS)])
    folder = ImageFolder(
        path,
        transform=transform)
    return DataLoader(
        folder,
        batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True)


STATS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def denorm(img):
    return img * STATS[1][0] + STATS[0][0]


def dump(
    model,
    device,
    index,
    latent,
    path,
    show=True
):
    img = model.G(latent).to(device)
    save_image(
        denorm(img),
        path / 'generated-{0:0=4d}.png'.format(index),
        nrow=8)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(img.cpu().detach(), nrow=8).permute(1, 2, 0))