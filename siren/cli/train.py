import enum
import sys
import fire
import torch
from traceback import format_exc
from pathlib import Path
from siren.data.utils import load_cats, load_mnist
from siren.models.dcgan import Model as DCGAN
from siren.models.sirengan import Model as SIRENGAN
from siren.models.train import fit


class Target(str, enum.Enum):
    DCGAN = 'dcgan'
    SIRENGAN = 'sirengan'


class Dataset(str, enum.Enum):
    CATS = 'cats'
    MNIST = 'mnist'


def train(target, dataset, path_to_data, path_to_dump, path_to_model, path_to_processed=None):
    """Trains GAN models.
    Args:
        target (str): The target model to train.
        dataset (str): The target dataset.
        path_to_data (str): The path to the data directory.
        path_to_dump (str): The path to the data dump directory.
        path_to_model (str): The path to the model directory.
    """
    path_to_data = Path(path_to_data).expanduser()
    path_to_dump = Path(path_to_dump).expanduser()
    path_to_dump.mkdir(parents=True, exist_ok=True)
    path_to_model = Path(path_to_model).expanduser()
    path_to_model.mkdir(parents=True, exist_ok=True)
    if path_to_processed:
        path_to_processed = Path(path_to_processed).expanduser()
        path_to_processed.mkdir(parents=True, exist_ok=True)
    model = get_model(target, dataset)
    loader = get_loader(dataset, path_to_data, path_to_processed)
    fit(
        model,
        get_device(),
        loader,
        path_to_dump,
        epochs=10)
    model.save(path_to_model)


def get_model(target, dataset, **kargs):
    options = {}
    if dataset == Dataset.CATS:
        options = { 'channels': 3, 'dim': 64 }
    elif dataset == Dataset.MNIST:
        options = { 'channels': 1, 'dim': 64 }
    if target == Target.DCGAN:
        print('DCGAN model specified')
        return DCGAN(**options)
    elif target == Target.SIRENGAN:
        print('SIRENGAN model specified')
        return SIRENGAN(**options)
    print('Default model (DCGAN) specified')
    return DCGAN(**options)


def get_loader(dataset, path_to_data, path_to_processed, **kargs):
    if dataset == Dataset.CATS:
        print('CATS dataset specified')
        return load_cats(
            path_to_data,
            **kargs)
    elif dataset == Dataset.MNIST:
        print('MNIST dataset specified')
        return load_mnist(
            path_to_data,
            path_to_processed,
            **kargs)
    print('Default dataset (CATS) specified')
    return load_cats(
        path_to_data,
        **kargs)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


def main():
    try:
        fire.Fire(train)
    except Exception:
        print(format_exc(), file=sys.stderr)
