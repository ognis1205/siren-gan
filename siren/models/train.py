import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision.datasets import ImageFolder
from pathlib import Path


STATS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def new_data_loader(
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


def train_g(
    model,
    opt_g, 
    device,
    batch_size = 128,
    latent_size = 128
):
    opt_g.zero_grad()
    latent = torch.randn(batch_size, latent_size, 1, 1).to(device)
    imgs = model.G(latent).to(device)
    preds = model.D(imgs).to(device)
    targets = torch.ones(batch_size, 1).to(device)
    loss = F.binary_cross_entropy(preds, targets)
    loss.backword()
    opt_g.step()
    return loss.item(), latent


def train_d(
    model,
    opt_d,
    device,
    real_imgs,
    batch_size = 128,
    latent_size = 128
):
    opt_d.zero_grad()
    real_preds = model.D(real_imgs).to(device)
    real_targets = torch.ones(real_imgs.size(0), 1).to(device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    latent = torch.randn(batch_size, latent_size, 1, 1).to(device)
    fake_imgs = model.G(latent).to(device)
    fake_targets = torch.zeros(fake_imgs.size(0), 1).to(device)
    fake_preds = model.D(fake_imgs).to(device)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()
    loss = real_loss + fake_loss
    loss.backword()
    opt_d.step()
    return loss.item(), real_score, fake_score


def fit(
    model,
    device,
    data_loader,
    path,
    epochs = 100,
    lr = 0.0002,
    start_index = 1
):
    torch.cuda.empty_cache()
    losses_g, losses_d = [], []
    real_scores, fake_scores = [], []
    opt_d = torch.optim.Adam(
        model.D.to(device).parameters(),
        lr=lr,
        betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(
        model.G.to(device).parameters(),
        lr=lr,
        betas=(0.5, 0.999))
    for epoch in range(epochs):
        for real_imgs, _ in tqdm(data_loader):
            real_imgs = real_imgs.to(device)
            loss_d, real_score, fake_score = train_d(
                model,
                opt_d,
                device,
                real_imgs)
            loss_g, latent = train_g(
                model,
                opt_g,
                device)
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        print(
            f'Epoch [{epoch + 1}/{epochs}], '
            f'loss_g: {loss_g}, '
            f'loss_d: {loss_d}, '
            f'real_score: {real_score}, '
            f'fake_score: {fake_score}')
        dump(
            model,
            devive,
            epoch + start_index,
            latent,
            path,
            show=False)
