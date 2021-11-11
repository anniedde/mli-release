import os
import itertools
import argparse
import json
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib.lines import Line2D
import tqdm
import sys

sys.path.insert(1, '/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release')
from lib.mli.data import load_data
from utils import get_model, get_run_model_states, interp_networks, load_model_and_data
from lib.mli_eval.model.loss import EvalClassifierLoss

parser = argparse.ArgumentParser()
parser.add_argument("rundir")
parser.add_argument("outdir")
parser.add_argument("-d", "--data_eval_size", type=int, default=None)
parser.add_argument("-a", "--alphas", type=int, default=50)
parser.add_argument("--random_states", type=int, default=10)
parser.add_argument("--init_scale", type=float, default=1.0)
parser.add_argument("--show", action='store_true')
args = parser.parse_args()


def random_state_dict(model_state):
    """Returns a new state dictionary filled with random values
    """
    state = {}
    for p_name in model_state:
      # copy the state
      state[p_name] = model_state[p_name].clone().detach()
      # reinitialize if weight or bias
      if 'bias' in p_name:
        # using fanout, as hard to get fanin reliably
        bound = args.init_scale / np.sqrt(state[p_name].shape[0])
        nn.init.uniform_(state[p_name], -bound, bound)
      if 'weight' in p_name:
        nn.init.kaiming_uniform_(state[p_name], a=np.sqrt(5))
        state[p_name] *= args.init_scale
    return state
        
def randomly_perturb_state(model_state, stddev=1):
    state = {}
    for p_name in model_state:
      # copy the state
      state[p_name] = model_state[p_name].clone().detach()
      # Noise the weights and biases
      if 'bias' in p_name or 'weight' in p_name:
        state[p_name] += torch.randn_like(state[p_name]) * stddev
    return state


def compute_interp_data(model, loader, evalloader, init_state, get_rand_state, final_state, outdir):
  #orig_alphas, metrics = interp_networks(model, init_state, final_state, loader, [evalloader], args.alphas, True)
  #orig_losses = np.array(metrics[0]['loss'])
  # From random initialization
  alphas = None
  losses = []
  for _ in range(args.random_states):
    # Get a new random model
    random_state = get_rand_state()
    alphas, metrics = interp_networks(model, random_state, final_state, loader, [evalloader], args.alphas, EvalClassifierLoss(), cuda=True)
    losses.append(
      metrics[0]['loss']
    )
  data = {}
  data["losses"] = losses
  with open(os.path.join(outdir, 'randominit-final-interpolation.json'), 'w') as outfile:
    json.dump(data, outfile)
  rand_losses = np.array(losses)
  #np.save(os.path.join(outdir, 'alphas'), orig_alphas)
  #np.save(os.path.join(outdir, 'rand_losses'), rand_losses)
  #np.save(os.path.join(outdir, 'orig_losses'), orig_losses)
  return rand_losses

if __name__ == '__main__':

  rundirs = [x[0] for x in os.walk(args.rundir)]
  rundirs.remove(args.rundir)
  #print(rundirs)
  for rundir in rundirs:
    print(rundir)

    run_states = get_run_model_states(rundir)
    config = run_states['config']
    model_name = config['model_name']
    num_classes = config['num_classes'] if 'num_classes' in config else 10
    dset_name = config['dset_name']
    identity_init = False#config['identity_init'] if 'identity_init' in config else False
    batchsize = 128
    datasize = 10000#config['datasize']
    evalsize = datasize if not args.data_eval_size else args.data_eval_size
    steps = args.alphas

    model, loader = load_model_and_data(
      model_name, num_classes, dset_name, batchsize, datasize, True, False, False, identity_init
    )
    evalloader = load_data(dset_name, batchsize, evalsize, train=True, shuffle=None, random_augment_train=True)
    model.cuda()
    outdir = args.outdir
    try:
      os.makedirs(outdir)
    except:
      pass
    
    init_state = run_states['init_state']
    final_state = run_states['final_state']

    alpha_path = os.path.join(outdir, 'alphas.npy')
    orig_losses_path = os.path.join(outdir, 'orig_losses.npy')
    rand_losses_path = os.path.join(outdir, 'rand_losses.npy')

    if os.path.isfile(alpha_path) and os.path.isfile(rand_losses_path) and os.path.isfile(orig_losses_path):
      alphas = np.load(alpha_path)
      orig_losses = np.load(orig_losses_path)
      rand_losses = np.load(rand_losses_path)
    else:
      rand_losses = compute_interp_data(
        model,
        loader,
        evalloader,
        init_state,
        lambda: get_model(model_name, num_classes, identity_init).cuda().state_dict(),
        final_state,
        rundir
      )
    
    plt.figure()

    plt.plot(np.linspace(0, 1, steps, endpoint=True), rand_losses, label='Interpolation Loss')
    plt.xlabel(r"Interpolation ($\alpha$)")
    plt.ylabel("Train Loss")
    plt.xlim(0, 1)

    run_num = config["run_num"]
    fname_labels = ["dset_name", "model_name", "optim_name", "lr"]
    fname = ""
    if len(fname_labels) > 0:
      fname += ",".join(["{}={}".format(l, str(config[l])) for l in fname_labels])
    fname += ",run_num={},randomInit".format(run_num)
    fpath = os.path.join(outdir, fname)
    plt.title(fname, wrap=True)
    plt.tight_layout()
    plt.savefig(fpath + ".png", dpi=300, bbox_inches = 'tight')
    plt.clf()
    plt.close()

