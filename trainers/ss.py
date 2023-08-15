import time
import torch
import tqdm
import torch.nn as nn
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
import torch.nn.functional as F
import numpy as np

__all__ = ["train", "validate", "modifier"]


def train(train_loader, targets_network, groups_network, criterion, targets_network_optimizer, groups_network_optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    #unfairness = AverageMeter("Unfairness", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    targets_network.train()
    groups_network.train()
    end = time.time()

    eopp_list = torch.zeros(2, 10).cuda()
    data_count = torch.zeros(2, 10).cuda()
    for i, (images, targets, groups) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
            groups = groups.cuda(args.gpu, non_blocking=True).to(torch.int64)

        #compute output
        targets_outputs = targets_network(images)
        #groups_outputs = groups_network(targets_outputs)
        targets_loss = criterion(targets_outputs, targets)

        #groups_loss = criterion(groups_outputs, groups)
        #print(groups_network.parameters())
        #groups_grad = torch.autograd.grad(groups_loss, filter(lambda p:p.requires_grad, groups_network.parameters()),#groups_network.parameters(),
        #                                    retain_graph=True, allow_unused=True)
        #for param, grad in zip(filter(lambda p:p.requires_grad, groups_network.parameters()), groups_grad):
        #    param.grad = grad

        #if epoch % args.adv_training_ratio == 0:
        #    grad_from_targets = torch.autograd.grad(targets_loss, filter(lambda p:p.requires_grad, targets_network.parameters()),
        #                            retain_graph=True, allow_unused=True)
        #    grad_from_groups = torch.autograd.grad(groups_loss, filter(lambda p:p.requires_grad, targets_network.parameters()),
        #                            retain_graph=True, allow_unused=True)
        #    for param, targets_grad, groups_grad in zip(filter(lambda p:p.requires_grad, targets_network.parameters()), grad_from_targets, 
        #s                                                grad_from_groups):
                # Gradient projection
        #        if groups_grad.norm() > 1e-5:
        #            param.grad = targets_grad - args.alpha*groups_grad - \
        #                    ((targets_grad*groups_grad).sum()/groups_grad.norm()) \
        #                    * (groups_grad/groups_grad.norm()) 
        #        else:
        #            param.grad = targets_grad - args.alpha*groups_grad 
        #targets_network_optimizer.step()
        #groups_network_optimizer.step()
        targets_network_optimizer.zero_grad()
        targets_loss.backward()
        targets_network_optimizer.step()


        _, preds = targets_outputs.max(1)
        acc = (targets ==  preds).float()
        acc1 = (targets ==  preds).float().mean()
        for g in range(2):
            for l in range(10):
                eopp_list[g, l] += acc[(groups == g) * (targets == l)].sum()
                data_count[g, l] += torch.sum((groups == g) * (targets == l))

        losses.update(targets_loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        #unfairness.update(max_eopp.item(), images.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)
    print(eopp_list)
    print(data_count)
    eopp_list = eopp_list / data_count
    max_eopp = torch.max(eopp_list, dim=0)[0] - torch.min(eopp_list, dim=0)[0]
    max_eopp = torch.max(max_eopp)
    print("\n",'max_eopps',max_eopp)
    return top1.avg, max_eopp, losses.avg


       


def validate(val_loader, targets_network, groups_network, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    #unfairness = AverageMeter("Unfairness", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1], prefix="Test: "
    )

    # switch to evaluate mode
    targets_network.eval()
    groups_network.eval()

    with torch.no_grad():
        end = time.time()
        eopp_list = torch.zeros(2, 10).cuda()
        data_count = torch.zeros(2, 10).cuda()

        for i, (images, targets, groups) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
                groups = groups.cuda(args.gpu, non_blocking=True).to(torch.int64)

            targets_outputs = targets_network(images)
            groups_outputs = groups_network(targets_outputs)

            targets_loss = criterion(targets_outputs, targets)
            groups_loss = criterion(groups_outputs, groups)




            _, preds = targets_outputs.max(1)
            acc = (targets ==  preds).float()
            acc1 = (targets ==  preds).float().mean()

            for g in range(2):
                for l in range(10):
                    eopp_list[g, l] += acc[(groups == g) * (targets == l)].sum()
                    data_count[g, l] += torch.sum((groups == g) * (targets == l))


            losses.update(targets_loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            #unfairness.update(max_eopp.item(), images.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    print(eopp_list)
    print(data_count)
    eopp_list = eopp_list / data_count
    max_eopp = torch.max(eopp_list, dim=0)[0] - torch.min(eopp_list, dim=0)[0]
    max_eopp = torch.max(max_eopp)
    print("\n",'max_eopps',max_eopp)
    return top1.avg, max_eopp, losses.avg

def modifier(args, epoch, targets_network, groups_network):
    return
