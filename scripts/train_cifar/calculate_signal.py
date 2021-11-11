import functools
import sys, os

import json
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from munch import Munch

import torch

sys.path.insert(1, '/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release')
import lib.mli.models as models
from lib.mli.data import load_data, load_data_subset
from lib.mli.models import interpolate_state

alpha = 0.5

path = "/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/output_signal_data/vgg19-noBN-sgd-lr0_01-alpha{}-expectation/".format(str(alpha).replace(".", "_"))
interpolated_output = torch.load(os.path.join(path, 'interpolated_output.pt'), map_location=torch.device('cpu'))
final_output = torch.load(os.path.join(path, 'final_output.pt'), map_location=torch.device('cpu'))

signals = []
layers = [1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]
for layer in layers:

    interpolated_layer_output = torch.flatten(interpolated_output['features[{}]'.format(layer)])
    interpolated_layer_output = torch.div(interpolated_layer_output, torch.norm(interpolated_layer_output))
    final_layer_output = torch.flatten(final_output['features[{}]'.format(layer)])
    final_layer_output = torch.div(final_layer_output, torch.norm(final_layer_output))

    signal = (torch.dot(interpolated_layer_output, final_layer_output)) / (torch.norm(final_layer_output)**2)
    signals.append(signal.item())
    print(signal.item())

#path = "/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/output_signal_data/vgg19-noBN-sgd-lr0_01-alpha0_5-expectation/"
isExist = os.path.exists(path)

if not isExist:
  
  # Create a new directory because it does not exist 
  os.makedirs(path)
  print("The new directory is created!")

save_dict = {}
save_dict['layers'] = layers
save_dict['signals'] = signals

plt.scatter([1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35], signals)
plt.title('Layer in Network vs. Signal Ratio (of expectations)')
plt.xlabel('Layer in VGG 19 Network')
plt.ylabel('Projection of Expected Interpolated Output on Expected Final Output', wrap=True)

plt.savefig(os.path.join(path, 'signals-temp.png'), dpi=300)

with open(os.path.join(path, 'signals-temp.json'), 'w') as f:
    json.dump(save_dict, f)