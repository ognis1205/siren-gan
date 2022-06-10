import torch


def generate(model, z, dim, number_of_images):
    samples = model.G(z).data.cpu().numpy()[:number_of_images]
    generated = []
    for sample in samples:
        if model.channels == 3:
            generated.append(sample.reshape(model.C, dim, dim))
        else:
            generated.append(sample.reshape(dim, dim))
    return generated


def walk(model, number_of_images):
    pass
