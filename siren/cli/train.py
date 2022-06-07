import enum
import sys
import fire
import torch
from traceback import format_exc
from pathlib import Path
from siren.data.utils import load
from siren.models.dcgan import Model as DCGAN
from siren.models.train import fit


class Target(enum.Enum):
    DCGAN = 'dcgan'


def train(target, path_to_data, path_to_dump, path_to_model):
    """Trains GAN models.
    Args:
        target (str): The target model to train.
        path_to_data (str): The path to the data directory.
        path_to_dump (str): The path to the data dump directory.
        path_to_model (str): The path to the model directory.
    """
    model = get_model(target)
    fit(
        model,
        get_device(),
        load(Path(path_to_data).expanduser()),
        Path(path_to_dump).expanduser())
    model.save(Path(path_to_model).expanduser())


def get_model(target, **kargs):
    if target == Target.DCGAN:
        return DCGAN(**kargs)
    return DCGAN(**kargs)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


def main():
    try:
        fire.Fire(train)
    except Exception:
        print(format_exc(), file=sys.stderr)
