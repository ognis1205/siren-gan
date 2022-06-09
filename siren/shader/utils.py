import re
import numpy as np


def dump(tensor):
  tensor = tensor.cpu().detach().numpy()
  return tensor


def vec4(ws):
  vec = 'vec4(' + ','.join([f'{w:.4g}' for w in ws]) + ')'
  vec = re.sub(r'\b0\.', '.', vec)
  return vec


def mat4(ws):
  mat = 'mat4(' + ','.join([f'{w:.4g}' for w in np.transpose(ws).flatten()]) + ')'
  mat = re.sub(r'\b0\.', '.', mat)
  return mat


def serialize(siren, var):
    # layer 1.
    omega = siren.omega
    chunks = int(siren.hidden_)
