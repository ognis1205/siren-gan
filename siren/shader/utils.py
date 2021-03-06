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
        y = in_w[row * 4:(row + 1) * 4, 1] * -1 * omega
        z = in_w[row * 4:(row + 1) * 4, 2] * omega
        w = in_w[row * 4:(row + 1) * 4, 3] * omega
        b = in_b[row * 4:(row + 1) * 4] * omega
        print(
            f'vec4 {var}0_{row} = sin(\n    '
#            f'uv.x * {vec4(x)} + \n    '
#            f'uv.y * {vec4(y)} + \n    '
#            f'z.x * {vec4(z)} + \n    '
#            f'z.y * {vec4(w)} + \n    '
            f'p.x * {vec4(x)} + \n    '
            f'p.y * {vec4(y)} + \n    '
            f'p.z * {vec4(z)} + \n    '
            f'p.w * {vec4(w)} + \n    '
            f'{vec4(b)});')

    # hidden layers.
    for layer in range(siren.hidden_layers):
        layer_w = dump(siren.main[layer + 1].linear.weight)
        layer_b = dump(siren.main[layer + 1].linear.bias)
        for row in range(chunks):
            line = f'vec4 {var}{layer + 1}_{row} = sin(\n    '
            line += ' +\n    '.join([
                f'{mat4(layer_w[row * 4:(row + 1) * 4, col * 4:(col + 1) * 4] * omega)}'
                f' * {var}{layer}_{col}'
                for col in range(chunks)])
            line += f' +\n    {vec4(layer_b[row * 4:(row + 1) * 4] * omega)});'
            print(line)

    # output layer.
    out_w = dump(siren.main[-2].weight[0])
    out_b = dump(siren.main[-2].bias[0])
    line = f'float {var} = \n    '
    for row in range(chunks):
        line += f'dot({var}{siren.hidden_layers}_{row}, {vec4(out_w[row * 4:(row + 1) * 4] * .5)}) + \n    '
    line += f'{out_b * .5 + .5:0.3f};'
    print(line)

#    color = ['r', 'g', 'b']
#    for i in range(3):
#        out_w = dump(siren.main[-2].weight[i])
#        out_b = dump(siren.main[-2].bias[i])
#        line = f'float {color[i]} = \n    '
#        for row in range(chunks):
#            line += f'dot({var}{siren.hidden_layers}_{row}, {vec4(out_w[row * 4:(row + 1) * 4] * .5)}) + \n    '
#        line += f'{out_b * .5 + .5:0.3f};'
#        print(line)

    print(f'return vec4(tanh({var}), tanh({var}), tanh({var}), 1.0);')
