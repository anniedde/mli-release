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
import sys

checkpoint = torch.load('/usr/xtmp/CSPlus/VOLDNN/Annie/mli-release/runs/mli_cifar10/1572454/final.pt', map_location=torch.device('cpu'))
#start_epoch = checkpoint["epoch"]
#model.load_state_dict(checkpoint["model_state"])

print(checkpoint)