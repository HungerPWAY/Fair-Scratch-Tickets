import os
import pathlib
import random
import time

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
)

from args import args
import importlib

import data
import models
import pandas as pd




def main():
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    args.gpu = None
    test = get_trainer(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)

    model = set_gpu(args, model)

    checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")
    model.load_state_dict(checkpoint["state_dict"])

    data = get_dataset(args)

    if args.label_smoothing is None:
        #criterion = nn.CrossEntropyLoss().cuda()
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)


    # Set up directories

    attrs = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young '.split()
    ti = attrs.index("Smiling")
    si = attrs.index("Male")
    (Pmale, Pfem) = (data.test_dataset.attr[:,si].float().mean(), 1 - data.test_dataset.attr[:,si].float().mean())


    # Save the initial state

    # Start training

        # evaluate on validation set
    df = pd.DataFrame(columns = ["Pruning_Rate", 'Lam_fair', 'Accuracy', 'Unfairness'])

    acc1, acc5, running_unfair = test(data.test_loader, model, criterion, args, Pmale, Pfem)
    #f = open("test_result.txt", "a+")
    #f.write(f"{args.fair_regularization}_{args.lam_fair}_{args.prune_rate}_ACC@1:{acc1}_Unfairness{(running_unfair[2]/running_unfair[3] - running_unfair[0]/running_unfair[1])}\n")
    d = {"Pruning_Rate": args.prune_rate,
        "Lam_fair": args.lam_fair,
        "Accuracy": acc1, 
        "Unfairness": (running_unfair[2]/running_unfair[3] - running_unfair[0]/running_unfair[1]).item()
        }
    print(d)
    if os.path.exists(f"logistic_dp_resnet50_Pruning_Rate=0.05.csv"):
        df = pd.read_csv(f"logistic_dp_resnet50_Pruning_Rate=0.05.csv")
    df = df.append(d, ignore_index= True)
    #print(df)
    df.to_csv(f"logistic_dp_resnet50_Pruning_Rate=0.05.csv", index = False)
    print("ACC@1: ", acc1)
    print("ACC@5: ", acc5)
    print("Unfairness: ", (running_unfair[2]/running_unfair[3] - running_unfair[0]/running_unfair[1]))


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.test


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model

def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # applying sparsity to the network
    if (
        args.conv_type != "DenseConv"
        and args.conv_type != "SampleSubnetConv"
        and args.conv_type != "ContinuousSparseConv"
    ):
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")

        set_model_prune_rate(model, prune_rate=args.prune_rate)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    # freezing the weights if we are only doing subnet training
    if args.freeze_weights:
        freeze_model_weights(model)

    return model


if __name__ == "__main__":
    main()