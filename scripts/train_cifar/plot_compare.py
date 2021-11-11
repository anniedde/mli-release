import functools
import sys, os

import json
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from munch import Munch
import argparse
from copy import deepcopy
from math import sqrt

import torch

sys.path.insert(1, '/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release')
import lib.mli.models as models
from lib.mli.data import load_data, load_data_subset
from lib.mli.models import interpolate_state


with open('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/output_signal_data/vgg19-noBN-sgd-lr0_01-alpha0_5-trueSignal-run1/signals.json') as f:
    signals = json.load(f)
#layers = signals["layers"]
#signals = signals["signals"]

with open('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/output_signal_data/vgg19-noBN-sgd-lr0_01-alpha0_5-trueSignal-run1/norms.json') as f:
    norms = json.load(f)

norms_plot = []
for i, layer in enumerate(signals["layers"]):
    norms_plot.append(signals["signals"][i]/norms[layer])


layers = [1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]
layers = [elem - 1 for elem in layers]

plt.scatter(layers, norms_plot)
plt.title('Signal/Norm')
plt.xlabel('Layer in VGG 19 Network')
plt.ylabel('Signal/Norm', wrap=True)

plt.savefig('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/output_signal_data/vgg19-noBN-sgd-lr0_01-alpha0_5-trueSignal-run1/signals-norms.png', dpi=300)

#torch.save(final_output, os.path.join(path, 'final_output.pt'))
print('done!')