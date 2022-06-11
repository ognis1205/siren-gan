import torch
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from siren.data.utils import denorm


def generate(model, z, number_of_images, image_size = 64):
    samples = model.G(z).data.cpu().numpy()[:number_of_images]
    generated = []
    for sample in samples:
        if model.channels == 3:
            generated.append(sample.reshape(model.C, image_size, image_size))
        else:
            generated.append(sample.reshape(image_size, image_size))
    return generated


def walk(model, zx, z1, z2, number_of_images, path_to_dump, image_size = 64):
    if model.cuda_enabled:
        zx = zx.cuda()
        z1 = z1.cuda()
        z2 = z2.cuda()
    zx = Variable(zx)
    a = 1.0 / float(number_of_images)
    images = []
    for i in range(0, number_of_images):
        zx.data = z1 * a + z2 * (1.0 - a)
        a += a
        inception = denorm(model.G(zx))
        images.append(inception.view(model.channels, image_size, image_size).data.cpu())
    grid = make_grid(
        images,
        nrow=number_of_images)
    save_image(
        grid,
        path_to_dump / 'interpolated.png')
