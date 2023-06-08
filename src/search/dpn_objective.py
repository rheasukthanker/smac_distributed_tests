#from comet_ml import Experiment
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from src.loss.focal import FocalLoss
from src.utils.utils import AverageMeter, accuracy
from src.utils.utils_train import Network
import numpy as np
import pandas as pd
import random
import math
from src.utils.utils_train import Network, get_head
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
import argparse
import argparse
import os
import pickle
import time
from src.search.dpn107 import DPN
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.search.operations import *
import os
import pickle
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import math
device = torch.device("cuda")
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
def fairness_objective_dpn(config, seed, budget):
    with open("/work/dlclarge2/sukthank-tanglenas/SMAC3/src/search/config_dpn107_CosFace_sgd.yaml","r") as ymlfile:
        args = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = dotdict(args)
    args.epochs = int(budget)
    args.opt = config["optimizer"]
    args.head = config["head"]
    print(args)
    p_images = {
        args.groups_to_modify[i]: args.p_images[i]
        for i in range(len(args.groups_to_modify))
    }
    p_identities = {
        args.groups_to_modify[i]: args.p_identities[i]
        for i in range(len(args.groups_to_modify))
    }
    args.p_images = p_images
    args.p_identities = p_identities

    print("P identities: {}".format(args.p_identities))
    print("P images: {}".format(args.p_images))
    directory ="Checkpoints/Checkpoints_Layers_{}_LR_{}_Head_{}_Optimizer_{}/".format(str(str(config["edge1"])+str(config["edge2"])+str(config["edge3"])), config["lr"], config["head"],config["optimizer"])
    if not os.path.exists(directory):
       os.makedirs(directory)
    args.batch_size=64
    #dataloaders, num_class, demographic_to_labels_train, demographic_to_labels_val, _ = prepare_data(
    #    args)
    args.num_class = 7058
    args.num_workers = 10
    edges=[int(config['edge1']),int(config['edge2']),int(config['edge3'])]
    # Build model
    backbone = DPN(edges,num_init_features=128, k_r=200, groups=50, k_sec=(1,1,1,1), inc_sec=(20, 64, 64, 128),num_classes=0)
    input=torch.ones(4,3,32,32)
    output=backbone(input)
    args.embedding_size= output.shape[-1]
    head = get_head(args)
    train_criterion = FocalLoss(elementwise=True)
    head,backbone= head.to(device), backbone.to(device)
    backbone = nn.DataParallel(backbone)
    model = Network(backbone, head)
    if (config["optimizer"] == "Adam") or (config["optimizer"] == "AdamW"):
        args.lr=config["lr_adam"]
    if config["optimizer"] == "SGD":
        args.lr=config["lr_sgd"]
    print(args.lr)
    print(args.opt)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    #scheduler, num_epochs = create_scheduler(args, optimizer)
    model.to(device)
    epoch=0
    start = time.time()
    print('Start training')


    print("P identities: {}".format(args.p_identities))
    print("P images: {}".format(args.p_images))
    while epoch < int(budget):
        for c in range(2000):
            model.train()  # set to training mode
            meters = {}
            meters["loss"] = AverageMeter()
            meters["top5"] = AverageMeter()
            inputs = torch.randn([64, 3, 224, 224])
            labels = torch.randint(0, 7058, [64])
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs, reg_loss = model(inputs, labels)
            loss = train_criterion(outputs, labels) + reg_loss
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            meters["loss"].update(loss.data.item(), inputs.size(0))
            meters["top5"].update(prec5.data.item(), inputs.size(0))
            print("Epoch: {} Loss: {} Top5: {}".format(epoch, meters["loss"].avg, meters["top5"].avg))
        epoch=epoch+1
            #break
    return {
        "negacc": -meters["top5"].avg, 
        "loss": meters["loss"].avg,
        }