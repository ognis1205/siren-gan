import torch
import torch.nn.functional as F
from tqdm import tqdm
from siren.data.utils import dump


def fit(
    model,
    device,
    data_loader,
    path_to_dump,
    batch_size = 128,
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
        for x, _ in tqdm(data_loader):
            x = x.to(device)
            loss_d, real_score, fake_score = model.train_d(
                device,
                batch_size,
                opt_d,
                x)
            loss_g, latent = model.train_g(
                device,
                batch_size,
                opt_g)
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
            device,
            epoch + start_index,
            latent,
            path_to_dump,
            show=False)
