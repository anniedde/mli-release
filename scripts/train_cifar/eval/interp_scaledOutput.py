# interpolate from initial to final point


import copy
import functools
import os
import sys

import json
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sacred import Experiment
from torch.optim.lr_scheduler import MultiStepLR

from munch import Munch

sys.path.insert(1, '/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release')
import lib.mli.models as models
from lib.mli.data import load_data
from lib.mli.metrics import param_dist
from lib.mli.models import get_loss_fn, interpolate_state
from lib.mli.metrics.gauss_len import compute_avg_gauss_length, compute_avg_gauss_length_bn
from lib.mli.optim import get_optimizer
from lib.mli_eval.plotting.interp import plot_interp
from lib.mli.sacred import SlurmFileStorageObserver

from lib.mli_eval.model.interp import interp_networks, interp_networks_outputScaled
from lib.mli_eval.model.loss import EvalClassifierLoss
from lib.mli_eval.model.utils import SumMetricsContainer


def get_config():
    # Data Config
    dset_name = "cifar10"
    datasize = 60000
    batchsize = 128

    # Model Config
    model_name = "alexnet"
    loss_fn = "ce"
    num_classes = 10
    identity_init = True

    # Initialization
    init_type = "default"

    # Optimizer Config
    epochs = 200
    optim_name = "sgd"
    lr = 0.1
    beta = 0.9
    weight_decay = 1e-4
    decay_milestones = [60, 90, 120]
    decay_factor = 0.1

    # Misc
    alpha_steps = 50
    cuda = True
    min_loss_threshold = 2.3
    min_loss_epoch_check = 50  # Before first lr decay by default
    log_wdist = True

    # Experiment Config
    tag = "cifar10"
    seed = 17
    save_freq = 25
    eval_gl = False
    run_num = 1


def load_data_captured(dset_name, batchsize, datasize, train=True):
    return load_data(dset_name, batchsize, datasize, train)


def get_optimizer_captured(parameters, optim_name, lr, beta, weight_decay, decay_milestones, decay_factor):
    optimizer = get_optimizer(optim_name, lr, beta, weight_decay=weight_decay)(parameters)
    lr_scheduler = MultiStepLR(optimizer, decay_milestones, decay_factor, -1)
    return optimizer, lr_scheduler


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


def compute_loss(model, out, targets, loss_fn):
    return get_loss_fn(loss_fn)(out, targets)


def warm_bn(model, loader, cuda):
    model.reset_bn()
    model.train()
    with torch.no_grad():
        for x, y in loader:
            if cuda:
                x, y = x.cuda(), y.cuda()
            _logits = model(x)



with open('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/runs/mli_cifar10/1577604/config.json') as f:
    config = json.load(f)

cfg = Munch.fromDict(config)
train_loader = load_data(cfg.dset_name, cfg.batchsize, cfg.datasize, train=True)

model = get_model(cfg.model_name, cfg.num_classes, cfg.identity_init)
if cfg.cuda:
    model = model.cuda()

# load init and final states
init_state = torch.load('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/runs/mli_cifar10/1577604/init.pt')
init_state = init_state["model_state"]
final_state = torch.load('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/runs/mli_cifar10/1577604/final.pt')
final_state = final_state["model_state"]

# Evaluate interpolation path of networks
alphas, metrics = interp_networks_outputScaled(
    model, init_state, final_state, 
    train_loader, [train_loader],
    #cfg.alpha_steps, 
    20,
    EvalClassifierLoss(), cfg.cuda
)

with open('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/random_interpolation_test/1577604-alpha20.json', 'w') as json_file:
    json.dump(metrics[0], json_file)