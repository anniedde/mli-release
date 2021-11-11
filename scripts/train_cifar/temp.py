import functools
import sys, os

import json
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from munch import Munch
import argparse

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



batchsize = 1
datasize = 10000

# vgg-19, no bn, SGD, lr 0.01
#run_num = 1577604
#layers = [1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]

# resnet 32, no bn, SGD, lr 0.01, run num 2
run_num = 1575674
#layers = [1]

with open('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/runs/mli_cifar10/{}/config.json'.format(run_num)) as f:
    config = json.load(f)

cfg = Munch.fromDict(config)
train_loader = load_data(cfg.dset_name, batchsize, datasize, train=True, shuffle=False, random_augment_train=False)

model = get_model(cfg.model_name, cfg.num_classes, cfg.identity_init)
#if cfg.cuda:
#    model = model.cuda()

sd = {}
for name, param in model.named_modules():
    print(name)
    if 'relu' in name or 'fc' in name:
        
        param.register_forward_hook(get_activation('name', sd))

#model.layer1.0..register_forward_hook(get_activation('features[{}]'.format(layer), final_output)