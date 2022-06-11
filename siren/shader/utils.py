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
    omega = siren.omega
    chunks = int(siren.hidden_features / 4)

    # layer 1.
    in_w = dump(siren.main[0].linear.weight)
    in_b = dump(siren.main[0].linear.bias)
    for row in range(chunks):
        x = in_w[row * 4:(row + 1) * 4, 0] * omega
        y = in_w[row * 4:(row + 1) * 4, 1] * omega
        z = in_w[row * 4:(row + 1) * 4, 2] * omega
        w = in_w[row * 4:(row + 1) * 4, 3] * omega
        b = in_b[row * 4:(row + 1) * 4] * omega
        print(
            f'vec4 {var}0_{row} = sin('
            f'uv.x * {vec4(x)} + '
            f'uv.y * {vec4(y)} + '
            f'z.x * {vec4(z)} + '
            f'z.y * {vec4(w)} + '
            f'{vec4(b)});')

    # hidden layers.
    for layer in range(siren.hidden_layers):
        layer_w = dump(siren.main[layer + 1].linear.weight)
        layer_b = dump(siren.main[layer + 1].linear.bias)
        for row in range(chunks):
            line = f'vec4 {var}{layer + 1}_{row} = sin('
            line += ' +\n'.join([
                f'{mat4(layer_w[row * 4:(row + 1) * 4, col * 4:(col + 1) * 4] * omega)}'
                f' * {var}{layer}_{col}'
                for col in range(chunks)])
            line += f' +\n {vec4(layer_b[row * 4:(row + 1) * 4] * omega)});'
            print(line)
