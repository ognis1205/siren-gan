from setuptools import setup, find_packages
from pkg_resources import parse_requirements


setup(
    name='siren',
    version='0.1.0',
    description='SIREN: SIREN Based Compositional Pattern Producing Network',
    author='Shingo OKAWA',
    python_requires='==3.8.*',
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'tqdm',
        'fire',
        'jupyter',
    ],
    entry_points={
        "console_scripts": [
            "download_url = siren.cli.download_url:main",
            "download_google_drive = siren.cli.download_google_drive:main",
            "train = siren.cli.train:main",
        ],
    },
    packages=find_packages(exclude=['test', 'test.*']))
