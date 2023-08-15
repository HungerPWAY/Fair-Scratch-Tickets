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
import torch.optim as optim
import numpy as np
from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    freeze_model_subnet,
    init_model_weight_with_score,
    init_subnet_model_weight_with_score,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
)
from utils.schedulers import get_policy


from args import args
import importlib
import logging
import data
import models
from utils.builder import get_builder
import shutil




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

    if args.adv is not True:
        raise ValueError
    
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir



    log = logging.getLogger(__name__)
    log_path = os.path.join(run_base_dir, 'log.txt')
    handlers = [logging.FileHandler(log_path, mode='a+'),
                logging.StreamHandler()]
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    log.info(args)

    train, validate, modifier = get_trainer(args)
    args.gpu = None
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)
    if args.random_ticket:
        for name, param in model.named_parameters():
            if name.endswith("scores"):
                flat_scores = param.flatten()
                new_scores = torch.randperm(len(flat_scores))
                param.data = new_scores.view_as(param)
    model = set_gpu(args, model)






    

    if args.task == 'search':
        freeze_model_weights(model)
    else:
        freeze_model_subnet(model)

    
    if args.task == 'ft_full':
        init_model_weight_with_score(model, prune_rate=args.prune_rate)
    elif args.task == 'ft_reinit':
        args.init = args.ft_init
        builder = get_builder()
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                builder._init_conv(module)


    elif args.task == "ft_subnet_inherit":
        init_subnet_model_weight_with_score(model, prune_rate=args.prune_rate)
    elif args.task is None:
        raise ValueError('Task is not given!')
    elif args.task != "search" and args.task!= "dense":
        raise ValueError('The task is beyond our implementation!')
    
    
    targets_network_optimizer = get_optimizer(args, model)
    groups_network_optimizer = get_optimizer_adv(args, model)
    

    if args.pretrained:
        pretrained(args, model)




    



    if args.lr_policy_targets == "ReduceLROnPlateau":
        lr_policy_targets = optim.lr_scheduler.ReduceLROnPlateau(targets_network_optimizer, 'min', verbose=True)

    else:
        lr_policy_targets = get_policy(args.lr_policy)(targets_network_optimizer, args)

    if args.lr_policy_targets == "ReduceLROnPlateau":
        lr_policy_groups = optim.lr_scheduler.ReduceLROnPlateau(groups_network_optimizer, 'min', verbose=True)

    else:
        lr_policy_groups = get_policy(args.lr_policy)(groups_network_optimizer, args)

    
    data = get_dataset(args)

    if args.loss == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif args.loss == "CrossEntropyLoss": 
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Two conflicting loss functions are used simultaneously!")

    
    if args.label_smoothing and args.loss == "CrossEntropyLoss":
        criterion = LabelSmoothing(smoothing=args.label_smoothing)
    elif args.label_smoothing and args.loss == "BCEWithLogitsLoss":
        raise ValueError("This loss fucntion havs not been implemented!")




    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_unfairness = 0.0
    best_equality_gap_0 = 0.0
    best_equality_gap_1 = 0.0
    best_parity_gap = 0.0
    best_train_acc1 = 0.0
    best_train_unfairness = 0.0
    best_train_equality_gap_0 = 0.0
    best_train_equality_gap_1 = 0.0
    best_train_parity_gap = 0.0


    if args.automatic_resume:
        args.resume = ckpt_base_dir / 'model_latest.pth'
        if os.path.isfile(args.resume):

            #best_acc1, best_unfairness = resume(args, targets_network, targets_network_optimizer, mode='targets_network')
            best_acc1, best_unfairness = resume(args, model, targets_network_optimizer, groups_network_optimizer)
        else:
            print('Train from scratch.')


    if not args.evaluate:
        args.Nmatrix = data.train_Nmatrix
    else:
        args.Nmatrix = data.test_Nmatrix

    print(args.Nmatrix)
    # Data loading code
    if args.evaluate:
        test_acc1, unfairness, loss , equality_gap_0, equality_gap_1, parity_gap= validate(
            data.test_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        log.info('Acc@1: %.2f, Unfairness: %.2f', test_acc1, unfairness)


        Test_write_result_to_csv(
            test_acc1=test_acc1,
            test_unfairness= unfairness,
            prune_rate=args.prune_rate,
            lr = args.lr,
            batch_size = args.batch_size,
            lr_policy_targets = args.lr_policy_targets,
            adv_lr = args.adv_lr,
            lr_policy_groups = args.lr_policy_groups,
            base_config=args.config,
            name=args.name,
            lambd = args.lambd,
            test_equality_gap_0 = equality_gap_0,
            test_equality_gap_1 = equality_gap_1,
            test_parity_gap = parity_gap,
            )
        return

    # Set up directories


    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None
    unfairness = None
    equality_gap_0 = None
    equality_gap_1 = None
    parity_gap = None

    # Save the initial state
    save_checkpoint(
        {
            "epoch": args.start_epoch or 0,
            "arch": args.arch,
            "network_state_dict":  model.state_dict(),
            "best_acc1": best_acc1,
            "best_unfair":best_unfairness,
            "best_equality_gap_0": best_equality_gap_0,
            "best_equality_gap_1": best_equality_gap_1,
            "best_parity_gap": best_parity_gap,
            "best_train_acc1": best_train_acc1,
            "best_train_unfair": best_train_unfairness,
            "best_train_equality_gap_0": best_train_equality_gap_0,
            "best_train_equality_gap_1": best_train_equality_gap_1,
            "best_train_parity_gap": best_train_parity_gap,
            "targets_network_optimizer": targets_network_optimizer.state_dict(),
            "groups_network_optimizer": groups_network_optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
            "curr_unfainess": unfairness if unfairness else "Not evaluated",
            "curr_equality_gap_0": equality_gap_0 if equality_gap_0 else "Not evaluated",
            "curr_equality_gap_1": equality_gap_1 if equality_gap_1 else "Not evaluated",
            "curr_parity_gap": parity_gap if parity_gap else "Not evaluated",
        },
        False,
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        if args.lr_policy_targets != "ReduceLROnPlateau":
            lr_policy_targets(epoch, iteration=None)

        if args.lr_policy_groups != "ReduceLROnPlateau":
            lr_policy_groups(epoch, iteration=None)
        modifier(args, epoch, model)

        cur_lr_targets = get_lr(targets_network_optimizer)
        cur_lr_groups = get_lr(groups_network_optimizer)

        # train for one epoch
        start_train = time.time()
        
        train_acc1, train_unfairness, loss,  train_equality_gap_0, train_equality_gap_1, train_parity_gap = train(
            data.train_loader, model, criterion, targets_network_optimizer,groups_network_optimizer, epoch, args,writer=writer)

        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set

        if epoch % args.val_every == 0 or epoch == args.epochs - 1: 
            start_validation = time.time()
            acc1, unfairness, val_loss, equality_gap_0, equality_gap_1, parity_gap = validate(data.val_loader, model, criterion, args, writer, epoch)
            validation_time.update((time.time() - start_validation) / 60)
            if args.lr_policy_targets == "ReduceLROnPlateau":
                lr_policy_targets.step(val_loss)

            if args.lr_policy_groups == "ReduceLROnPlateau":
                lr_policy_groups.step(val_loss)


            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_unfairness = unfairness
                best_equality_gap_0 = equality_gap_0
                best_equality_gap_1 = equality_gap_1
                best_parity_gap = parity_gap


            if train_acc1 > best_train_acc1:
                best_train_acc1 = train_acc1
                best_train_unfairness = train_unfairness
                best_train_equality_gap_0 = train_equality_gap_0
                best_train_equality_gap_1 = train_equality_gap_1
                best_train_parity_gap = train_parity_gap


            is_last = (epoch == args.epochs - 1)
            if is_best or is_last:
                if is_best:
                    log.info(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")
                save_checkpoint(
                    {
                        "epoch": args.start_epoch or 0,
                        "arch": args.arch,
                        "network_state_dict":  model.state_dict(),
                        "best_acc1": best_acc1,
                        "best_unfair":best_unfairness,
                        "best_equality_gap_0": best_equality_gap_0,
                        "best_equality_gap_1": best_equality_gap_1,
                        "best_parity_gap": best_parity_gap,
                        "best_train_acc1": best_train_acc1,
                        "best_train_unfair": best_train_unfairness,
                        "best_train_equality_gap_0": best_train_equality_gap_0,
                        "best_train_equality_gap_1": best_train_equality_gap_1,
                        "best_train_parity_gap": best_train_parity_gap,
                        "targets_network_optimizer": targets_network_optimizer.state_dict(),
                        "groups_network_optimizer": groups_network_optimizer.state_dict(),
                        "curr_acc1": acc1,
                        "curr_unfainess": unfairness,
                        "curr_equality_gap_0": equality_gap_0,
                        "curr_equality_gap_1": equality_gap_1,
                        "curr_parity_gap": parity_gap,
                    },
                    is_best,
                    is_last,
                    filename=ckpt_base_dir / f"epoch_{epoch}.state",
                    save=False,
                )
            #test_acc1, test_unfairness, test_loss = validate(data.test_loader, targets_network, groups_network, criterion, args, writer=None, epoch=args.start_epoch)
            #log.info('Test Acc@1: %.2f, Test Unfairness: %.2f', test_acc1, test_unfairness)

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        #only targets_network
        if args.conv_type == "SampleSubnetConv":
            count = 0
            sum_pr = 0.0
            for n, m in model.named_modules():
                if isinstance(m, SampleSubnetConv):
                    # avg pr across 10 samples
                    pr = 0.0
                    for _ in range(10):
                        pr += (
                            (torch.rand_like(m.clamped_scores) >= m.clamped_scores)
                            .float()
                            .mean()
                            .item()
                        )
                    pr /= 10.0
                    writer.add_scalar("pr/{}".format(n), pr, epoch)
                    sum_pr += pr
                    count += 1

            args.prune_rate = sum_pr / count
            writer.add_scalar("pr/average", args.prune_rate, epoch)

        writer.add_scalar("val/lr", cur_lr_targets, epoch)
        end_epoch = time.time()

    write_result_to_csv(
        best_acc1=best_acc1,
        best_unfairness=best_unfairness,
        best_equality_gap_0 = best_equality_gap_0,
        best_equality_gap_1 = best_equality_gap_1,
        best_parity_gap = best_parity_gap,
        best_train_acc1=best_train_acc1,
        best_train_unfairness=best_train_unfairness,
        best_train_equality_gap_0 = best_train_equality_gap_0,
        best_train_equality_gap_1 = best_train_equality_gap_1,
        best_train_parity_gap = best_train_parity_gap,
        curr_acc1=acc1,
        curr_unfairness=unfairness,
        curr_equality_gap_0 = equality_gap_0,
        curr_equality_gap_1 = equality_gap_1,
        curr_parity_gap = parity_gap,
        prune_rate=args.prune_rate,
        lr = args.lr,
        batch_size = args.batch_size,
        lr_policy_targets = args.lr_policy_targets,
        adv_lr = args.adv_lr,
        lr_policy_groups = args.lr_policy_groups,      
        base_config=args.config,
        name=args.name,
        lambd = args.lambd,
    )


    log_dir_new = 'logs/log_'+args.name
    if not os.path.exists(log_dir_new):
        os.makedirs(log_dir_new)
    
    shutil.copyfile(log_path, os.path.join(log_dir_new, 'log_'+args.task+'.txt'))


    ## Test process
    args.Nmatrix = data.test_Nmatrix
    model.load_state_dict(torch.load(ckpt_base_dir/'model_best.pth')["network_state_dict"], strict = True)
    test_acc1, test_unfairness, test_loss, test_equality_gap_0, test_equality_gap_1, test_parity_gap = validate(data.test_loader, model, criterion, args, writer=None, epoch=args.start_epoch)
        
    log.info('Test Acc@1: %.2f, Test Unfairness: %.2f', test_acc1, test_unfairness)


    Test_write_result_to_csv(
        test_acc1=test_acc1,
        test_unfairness=test_unfairness,
        prune_rate=args.prune_rate,
        lr = args.lr,
        batch_size = args.batch_size,
        lr_policy_targets = args.lr_policy_targets,
        adv_lr = args.adv_lr,
        lr_policy_groups = args.lr_policy_groups,
        base_config=args.config,
        name=args.name,
        lambd = args.lambd,
        test_equality_gap_0 = test_equality_gap_0,
        test_equality_gap_1 = test_equality_gap_1,
        test_parity_gap = test_parity_gap,
        )


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier


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


def resume(args, model, targets_network_optimizer, groups_network_optimizer):

    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]
        best_unfairness = checkpoint["best_unfairness"]

        model.load_state_dict(checkpoint["network_state_dict"])

        targets_network_optimizer.load_state_dict(checkpoint["targets_network_optimizer"])

        groups_network_optimizer.load_state_dict(checkpoint["groups_network_optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1, best_unfairness
    else:
        print(f"=> No checkpoint found at '{args.resume}'")


def pretrained(args, model):
    
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )[f"network_state_dict"]

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))
        raise ValueError("=> no pretrained weights found at '{}'".format(args.pretrained))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


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
            f"=> Rough estimate targets model params {sum(int(p.numel() * (args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    # freezing the weights if we are only doing subnet training
    #if args.freeze_weights:
        #freeze_model_weights(model)

    return model

def get_optimizer_adv(args, model):
    for n, v in model.module.adv_head.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.adv_optimizer == "sgd":
        parameters = list(model.module.adv_head.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.adv_weight_decay,
                },
                {"params": rest_params, "weight_decay": args.adv_weight_decay},
            ],
            args.adv_lr,
            momentum=args.adv_momentum,
            weight_decay=args.adv_weight_decay,
            nesterov=args.adv_nesterov,
        )
    elif args.adv_optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.module.adv_head.parameters()), lr=args.adv_lr
        )

    return optimizer



def get_optimizer(args, model):
    for n, v in model.module.encoder.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    for n, v in model.module.classifier.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)
    
    if args.optimizer == "sgd":
        parameters = list(model.module.encoder.named_parameters()) + list(model.module.classifier.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        parameters = list(model.module.encoder.parameters()) + list(model.module.classifier.parameters())
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, parameters), lr=args.lr
        )

    return optimizer

def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}_lambd={args.lambd}_adv"
        )
    else:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}_lambd={args.lambd}_adv"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    #if _run_dir_exists(run_base_dir):
    #    rep_count = 0
    #    while _run_dir_exists(run_base_dir / str(rep_count)):
    #        rep_count += 1
    #
    #    run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / f"results_fairness_{args.name}.csv"

    if not results.exists():
        results.write_text(
            "Date Finished,"
            "Base Config,"
            "Name,"
            "Batch Size,"
            "Learning Rate,"
            "LR Policy Targets,"
            "Adv Learning Rate,"
            "Lr Policy Groups,"
            "Prune Rate,"
            "lambd,"
            "Current Val Top 1,"
            "Current Val Unfairness,"
            "Current Val Equality Gap 0,"
            "Current Val Equality Gap 1,"
            "Current Val Parity Gap,"
            "Best Val Top 1,"
            "Best Val Unfairness,"
            "Best Val Equality Gap 0,"
            "Best Val Equality Gap 1,"
            "Best Val Parity Gap,"
            "Best Train Top 1,"
            "Best Train Unfairness,"
            "Best Train Equality Gap 0,"
            "Best Train Equality Gap 1,"
            "Best Train Parity Gap\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now},"
                "{base_config},"
                "{name},"
                "{batch_size},"
                "{lr},"
                "{lr_policy_targets},"
                "{adv_lr},"
                "{lr_policy_groups},"
                "{prune_rate},"
                "{lambd},"
                "{curr_acc1:.03f},"
                "{curr_unfairness:.03f},"
                "{curr_equality_gap_0:.03f},"
                "{curr_equality_gap_1:.03f},"
                "{curr_parity_gap:.03f},"
                "{best_acc1:.03f},"
                "{best_unfairness:.03f},"
                "{best_equality_gap_0:.03f},"
                "{best_equality_gap_1:.03f},"
                "{best_parity_gap:.03f},"
                "{best_train_acc1:.03f},"
                "{best_train_unfairness:.03f},"
                "{best_train_equality_gap_0:.03f},"
                "{best_train_equality_gap_1:.03f},"
                "{best_train_parity_gap:.03f}\n"
            ).format(now=now, **kwargs)
        )


def Test_write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / f"results_fairness_{args.name}_test.csv"

    if not results.exists():
        results.write_text(
            "Date Finished,"
            "Base Config,"
            "Name,"
            "Batch Size,"
            "Learning Rate,"
            "Lr Policy Targets,"
            "Adv Learning Rate,"
            "Lr Policy Groups,"
            "Prune Rate,"
            "lambd,"
            "Test Val Top 1,"
            "Test Val Unfairness,"
            "Test Val equality Gap 0,"
            "Test Val equality Gap 1,"
            "Test Val Parity Gap\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now},"
                "{base_config},"
                "{name},"
                "{batch_size},"
                "{lr},"
                "{lr_policy_targets},"
                "{adv_lr},"
                "{lr_policy_groups},"
                "{prune_rate},"
                "{lambd},"
                "{test_acc1:.03f},"
                "{test_unfairness:.03f},"
                "{test_equality_gap_0:.03f},"
                "{test_equality_gap_1:.03f},"
                "{test_parity_gap:.03f}\n"
            ).format(now=now, **kwargs)
        )



if __name__ == "__main__":
    main()
