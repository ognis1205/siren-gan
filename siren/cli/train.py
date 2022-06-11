import enum
import sys
import fire
import torch
from traceback import format_exc
from pathlib import Path
from siren.data.utils import load
from siren.models.dcgan import Model as DCGAN
from siren.models.sirengan import Model as SIRENGAN
from siren.models.train import fit


class Target(str, enum.Enum):
    DCGAN = 'dcgan'
    SIRENGAN = 'sirengan'


def train(target, path_to_data, path_to_dump, path_to_model):
    """Trains GAN models.
    Args:
        target (str): The target model to train.
        path_to_data (str): The path to the data directory.
        path_to_dump (str): The path to the data dump directory.
        path_to_model (str): The path to the model directory.
    """
    path_to_data = Path(path_to_data).expanduser()
    path_to_dump = Path(path_to_dump).expanduser()
    path_to_dump.mkdir(parents=True, exist_ok=True)
    path_to_model = Path(path_to_model).expanduser()
    path_to_model.mkdir(parents=True, exist_ok=True)
    model = get_model(target)
    fit(
        model,
        get_device(),
        load(path_to_data),
        path_to_dump,
        epochs=10)
    model.save(path_to_model)


def get_model(target, **kargs):
    if target == Target.DCGAN:
        print('DCGAN model specified')
        return DCGAN(**kargs)
    elif target == Target.SIRENGAN:
        print('SIRENGAN model specified')
        return SIRENGAN(**kargs)
    print('Default model (DCGAN) specified')
    return DCGAN(**kargs)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


def main():
    try:
        fire.Fire(train)
    except Exception:
        print(format_exc(), file=sys.stderr)
