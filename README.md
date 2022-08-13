SIREN-GAN
==============================

A PoC Project of the SIREN GAN implementation in GLSL: [Shadertoy](https://www.shadertoy.com/view/fsGyWG)

Implementation Notes
------------

The concept of this project is to demonstrate how to "serialize" SIREN-based neural network models as GLSL codes.
Hence, the further optimization of the model is out of the project scope, so the following hyper parameters are
likely not be optimal:

 - Batch size,
 - Number of epochs,
 - Latent space size,
 - etc.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
	│
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── setup.py           <- makes project pip installable (pip install -e .) so src can be imported


--------

References
------------

 - [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
 - [A Haphazard Tutorial for making Neural SDFs in Shadertoy](https://www.youtube.com/watch?v=8pwXpfi-0bU)
