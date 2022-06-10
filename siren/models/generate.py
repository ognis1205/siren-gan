import torch
from torchvision.utils import make_grid, save_image


def generate(model, z, dim, number_of_images):
    samples = model.G(z).data.cpu().numpy()[:number_of_images]
    generated = []
    for sample in samples:
        if model.channels == 3:
            generated.append(sample.reshape(model.C, dim, dim))
        else:
            generated.append(sample.reshape(dim, dim))
    return generated


def walk(model, number_of_images, path_to_dump):
    zx = torch.FloatTensor(1, model.latent_size, 1, 1)
    z1 = torch.randn(1, model.latent_size, 1, 1)
    z2 = torch.randn(1, model.latent_size, 1, 1)
    if model.cuda_enabled:
        zx = z_intp.cuda()
        z1 = z1.cuda()
        z2 = z2.cuda()
    zx = Variable(zx)
    images = []
    alpha = 1.0 / float(number_of_images + 1)
    for i in range(1, number_of_images + 1):
        zx.data = z1 * alpha + z2 * (1.0 - alpha)
        alpha += alpha
        inception = model.G(zx)
        inception = inception.mul(0.5).add(0.5)
        images.append(inception.view(model.channels, 32, 32).data.cpu())
    grid = make_grid(
        images,
        nrow=number_of_images)
    save_image(
        grid,
        path_to_dump / 'sirengan' / 'interpolated.png')
