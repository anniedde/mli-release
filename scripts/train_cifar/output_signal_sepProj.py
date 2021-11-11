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

MODEL_MAP = {
    "resnet-20": models.resnet20,
    "fixup_resnet-20": models.fixup_resnet20,
    "resnet-20-nobn": functools.partial(models.resnet20, use_batchnorm=False),
    "resnet-32": models.resnet32,
    "fixup_resnet-32": models.fixup_resnet32,
    "resnet-32-nobn": functools.partial(models.resnet32, use_batchnorm=False),
    "resnet-44": models.resnet44,
    "fixup_resnet-44": models.fixup_resnet44,
    "resnet-44-nobn": functools.partial(models.resnet44, use_batchnorm=False),
    "resnet-56": models.resnet56,
    "fixup_resnet-56": models.fixup_resnet56,
    "resnet-56-nobn": functools.partial(models.resnet56, use_batchnorm=False),
    "resnet-110": models.resnet110,
    "fixup_resnet-110": models.fixup_resnet110,
    "resnet-110-nobn": functools.partial(models.resnet110, use_batchnorm=False),
    "vgg16": models.vgg16_bn,
    "vgg16-nobn": models.vgg16,
    "vgg19": models.vgg19_bn,
    "vgg19-nobn": models.vgg19,
    "lenet": models.LeNet,
    "alexnet": models.AlexNet
}

def get_model(model_name, num_classes, identity_init):
    if "fixup" not in model_name and "resnet" in model_name:
        return MODEL_MAP[model_name](num_classes=num_classes, identity_init=identity_init)
    else:
        return MODEL_MAP[model_name](num_classes=num_classes)

def get_activation(name, out_dict):
    def hook(model, input, output):
        out_dict[name] = output.detach()
    return hook

def get_mean_activation(name, out_dict):
    def hook(model, input, output):
        if name not in out_dict.keys():
            out_dict[name] = output.detach()
        else:
            out_dict[name] += output.detach()
    return hook

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

def subtract(dict1, dict2):
    ret = deepcopy(dict1)
    for layer in ret:
        ret[layer] = torch.sub(dict1[layer], dict2[layer])
    return ret



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-alpha', type=float,
                    help='alpha * final + (1-alpha) * initial')
args = parser.parse_args()

alpha = args.alpha
batchsize = 1
datasize = 10000

# vgg-19, no bn, SGD, lr 0.01
#run_num = 1577604

run_num = 1577593 # vgg-19, no bn, SGD, lr 0.01 run 1

layers = [1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]
layers = [elem - 1 for elem in layers]

path = "/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/output_signal_data/vgg19-noBN-sgd-lr0_01-alpha{}-trueSignal-run1/".format(str(alpha).replace(".", "_"))
print('path is ', path)
isExist = os.path.exists(path)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path)
  print("The new directory is created!")

with open('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/runs/mli_cifar10/{}/config.json'.format(run_num)) as f:
    config = json.load(f)

cfg = Munch.fromDict(config)
train_loader = load_data(cfg.dset_name, batchsize, datasize, train=True, shuffle=False, random_augment_train=False)
getMean_loader = deepcopy(train_loader)

model_interpolated = get_model(cfg.model_name, cfg.num_classes, cfg.identity_init)
if cfg.cuda:
    model_interpolated = model_interpolated.cuda()
model_final = get_model(cfg.model_name, cfg.num_classes, cfg.identity_init)
if cfg.cuda:
    model_final = model_final.cuda()

init_state = torch.load('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/runs/mli_cifar10/{}/init.pt'.format(run_num))
init_state = init_state["model_state"]
final_state = torch.load('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/runs/mli_cifar10/{}/final.pt'.format(run_num))
final_state = final_state["model_state"]

interpolate_state(model_interpolated.state_dict(), init_state, final_state, alpha)
model_final.load_state_dict(final_state)

interpolated_mean_handles = []
final_mean_handles = []
interpolated_mean_output = {}
final_mean_output = {}

for layer in layers:
    interpolated_mean_handles.append(model_interpolated.features[layer].register_forward_hook(get_mean_activation('features[{}]'.format(layer), interpolated_mean_output)))
for layer in layers:
    final_mean_handles.append(model_final.features[layer].register_forward_hook(get_mean_activation('features[{}]'.format(layer), final_mean_output)))

# get mean single outputs
model_interpolated.eval()
model_final.eval()
with torch.no_grad():
    for x,y in getMean_loader:
        x,y = x.cuda(), y.cuda()
        __ = model_interpolated(x)
        __ = model_final(x)

# remove forward hooks
for handle in interpolated_mean_handles:
    handle.remove()
for handle in final_mean_handles:
    handle.remove()

for layer, layer_output in final_mean_output.items():
    final_mean_output[layer] = torch.div(layer_output, datasize)

for layer, layer_output in interpolated_mean_output.items():
    interpolated_mean_output[layer] = torch.div(layer_output, datasize)

#########################

interpolated_handles = []
final_handles = []

interpolated_single_output = {}
final_single_output = {}
for layer in layers:
    interpolated_handles.append(model_interpolated.features[layer].register_forward_hook(get_activation('features[{}]'.format(layer), interpolated_single_output)))
    final_handles.append(model_final.features[layer].register_forward_hook(get_activation('features[{}]'.format(layer), final_single_output)))

signals_dict = {}
final_output_mean_squared_norms = {}

model_interpolated.eval()
model_final.eval()
i = 0
with torch.no_grad():
    for x,y in train_loader:
        x,y = x.cuda(), y.cuda()
        interpolated_single_output.clear()
        final_single_output.clear()
        __ = model_interpolated(x)
        __ = model_final(x)
        print(i)
        i += 1

        interpolated_single_output = subtract(interpolated_single_output, interpolated_mean_output)
        final_single_output = subtract(final_single_output, final_mean_output)

        for layer in interpolated_single_output:
            interpolated_layer_output = torch.flatten(interpolated_single_output[layer])
            final_layer_output = torch.flatten(final_single_output[layer])
            signal = torch.dot(interpolated_layer_output, final_layer_output)

            if layer not in signals_dict.keys():
                signals_dict[layer] = signal.item()
                final_output_mean_squared_norms[layer] = torch.norm(final_layer_output)**2

            else:
                signals_dict[layer] += signal.item()
                final_output_mean_squared_norms[layer] += torch.norm(final_layer_output)**2


for key in signals_dict:
    signals_dict[key] = signals_dict[key] / datasize
    final_output_mean_squared_norms[key] = sqrt(final_output_mean_squared_norms[key] / datasize)
    signals_dict[key] = signals_dict[key] / final_output_mean_squared_norms[key]

save_dict = {}
dict_layers = list(signals_dict.keys())
save_dict['layers'] = dict_layers

signals = []
for layer in dict_layers:
    signals.append(signals_dict[layer])
save_dict['signals'] = signals

torch.save(signals_dict, os.path.join(path, 'true_signals.pt'))

plt.scatter(layers, signals)
plt.title('Layer in Network vs. True Signal Ratio, alpha={}'.format(alpha))
plt.xlabel('Layer in VGG 19 Network')
plt.ylabel('Signal Ratio', wrap=True)
plt.savefig(os.path.join(path, 'signals.png'), dpi=300)

with open(os.path.join(path, 'signals.json'), 'w') as f:
    json.dump(save_dict, f)

#torch.save(final_output, os.path.join(path, 'final_output.pt'))
print('done!')