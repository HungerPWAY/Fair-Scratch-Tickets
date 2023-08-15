import time
import torch
import tqdm
import torch.nn as nn
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
import torch.nn.functional as F
import numpy as np
from utils.fairness_utils import calculateGenderConfusionMatrices, calculateEqualityGap, calculateParityGap

__all__ = ["train", "validate", "modifier"]


def train(train_loader, model, criterion, targets_network_optimizer, groups_network_optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.4f")
    data_time = AverageMeter("Data", ":6.4f")
    losses = AverageMeter("Loss", ":6.4f")
    top1 = AverageMeter("Acc@1", ":6.4f")
    unfairness = AverageMeter("Unfairness", ":6.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, unfairness],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    model.train()
    end = time.time()

    #eopp_list = torch.zeros(2, 10).cuda()
    #data_count = torch.zeros(2, 10).cuda()
    cm_m = None
    cm_f = None
    for i, (images, targets, groups, protected_labels) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True).to(torch.float)
            groups = groups.cuda(args.gpu, non_blocking=True).to(torch.float)
            protected_labels = protected_labels.cuda(args.gpu, non_blocking = True).to(torch.bool)
        #print(protected_labels)
        #compute output
        
        targets_outputs, (a , a_detached) = model(images, protected_labels)

        #print(targets_outputs.shape)
        targets_loss = criterion(targets_outputs, targets.view(-1,1))
        if a!=None:
            groups_loss = criterion(a, groups[protected_labels])

            loss = targets_loss - args.lambd*groups_loss
            #print(groups_loss)



            targets_network_optimizer.zero_grad()
            loss.backward()
            targets_network_optimizer.step()

            adversarial_loss = criterion(a_detached, groups[protected_labels])

            groups_network_optimizer.zero_grad()
            adversarial_loss.backward()

            groups_network_optimizer.step()
        else:
            loss = targets_loss
            targets_network_optimizer.zero_grad()
            loss.backward()
            targets_network_optimizer.step()



        
        #_, preds = targets_outputs.max(1)
        preds = (targets_outputs.reshape(-1) >= 0).float()

        acc = (targets ==  preds).float()
        acc1 = (targets ==  preds).float().mean()
        #for g in range(2):
        #    for l in range(10):
        #        eopp_list[g, l] += acc[(groups == g) * (targets == l)].sum()
        #        data_count[g, l] += torch.sum((groups == g) * (targets == l))
        sens_attr = groups.bool()
        sens_attr = sens_attr.reshape(-1)
        cur_unfairness = torch.tensor([preds[sens_attr].sum(), preds[ sens_attr].shape[0],
                               preds[~sens_attr].sum(), preds[~sens_attr].shape[0]])

        batch_cm_m, batch_cm_f = calculateGenderConfusionMatrices(preds.view(-1), targets.view(-1), sens_attr.view(-1))
        if cm_m is None and cm_f is None:
            cm_m = batch_cm_m
            cm_f = batch_cm_f
        else:
            cm_m = list(cm_m)
            cm_f = list(cm_f)
            for j in range(len(cm_m)):
                cm_m[j] += batch_cm_m[j]
                cm_f[j] += batch_cm_f[j]
            cm_m = tuple(cm_m)
            cm_f = tuple(cm_f)
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        unfairness.update((cur_unfairness[2]/cur_unfairness[3] - cur_unfairness[0]/cur_unfairness[1]).item(), images.size(0))  


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)
    
    avg_equality_gap_0, avg_equality_gap_1, attr_equality_gap_0, attr_equality_gap_1 = \
        calculateEqualityGap(cm_m, cm_f)
    avg_parity_gap, attr_parity_gap = calculateParityGap(cm_m, cm_f)
    s_eval = (', Equality Gap 0: %.4f, Equality Gap 1: %.4f, Parity Gap: %.4f') % (avg_equality_gap_0, avg_equality_gap_1, avg_parity_gap)
    print(s_eval)
    #print(eopp_list)
    #print(data_count)
    #eopp_list = eopp_list / data_count
    #max_eopp = torch.max(eopp_list, dim=0)[0] - torch.min(eopp_list, dim=0)[0]
    #max_eopp = torch.max(max_eopp)
    #print("\n",'max_eopps',max_eopp)
    return top1.avg, unfairness.avg, losses.avg, avg_equality_gap_0, avg_equality_gap_1, avg_parity_gap


       


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.4f", write_val=False)
    losses = AverageMeter("Loss", ":6.4f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.4f", write_val=False)
    unfairness = AverageMeter("Unfairness", ":6.4f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, unfairness], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    cm_m = None
    cm_f = None
    with torch.no_grad():
        end = time.time()
        #eopp_list = torch.zeros(2, 10).cuda()
        #data_count = torch.zeros(2, 10).cuda()

        for i, (images, targets, groups, protected_labels) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True).to(torch.float)
                groups = groups.cuda(args.gpu, non_blocking=True).to(torch.float)
                protected_labels = protected_labels.cuda(args.gpu, non_blocking = True).to(torch.bool)

            #targets_outputs = targets_network(images)
            #groups_outputs = groups_network(targets_outputs)

            #targets_loss = criterion(targets_outputs, targets)
            #groups_loss = criterion(groups_outputs, groups)



            #compute output

            targets_outputs, (a , a_detached) = model(images, protected_labels)


            targets_loss = criterion(targets_outputs, targets.view(-1,1))

            #groups_loss = criterion(a, groups[protected_labels])

            loss = targets_loss# - args.lambd*groups_loss

            #_, preds = targets_outputs.max(1)
            preds = (targets_outputs.reshape(-1) >= 0).float()
            
            acc = (targets ==  preds).float()
            acc1 = (targets ==  preds).float().mean()
            '''
            for g in range(2):
                for l in range(10):
                    eopp_list[g, l] += acc[(groups == g) * (targets == l)].sum()
                    data_count[g, l] += torch.sum((groups == g) * (targets == l))
            '''
            sens_attr = groups.bool()
            sens_attr = sens_attr.reshape(-1)
            cur_unfairness = torch.tensor([preds[sens_attr].sum(), preds[ sens_attr].shape[0],
                               preds[~sens_attr].sum(), preds[~sens_attr].shape[0]])

            batch_cm_m, batch_cm_f = calculateGenderConfusionMatrices(preds.view(-1), targets.view(-1), sens_attr.view(-1))
            if cm_m is None and cm_f is None:
                cm_m = batch_cm_m
                cm_f = batch_cm_f
            else:
                cm_m = list(cm_m)
                cm_f = list(cm_f)
                for j in range(len(cm_m)):
                    cm_m[j] += batch_cm_m[j]
                    cm_f[j] += batch_cm_f[j]
                cm_m = tuple(cm_m)
                cm_f = tuple(cm_f)
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            unfairness.update((cur_unfairness[2]/cur_unfairness[3] - cur_unfairness[0]/cur_unfairness[1]).item(), images.size(0))  
            #unfairness.update(unfairness.item(), images.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)


    avg_equality_gap_0, avg_equality_gap_1, attr_equality_gap_0, attr_equality_gap_1 = \
        calculateEqualityGap(cm_m, cm_f)
    avg_parity_gap, attr_parity_gap = calculateParityGap(cm_m, cm_f)
    s_eval = ('Equality Gap 0: %.4f, Equality Gap 1: %.4f, Parity Gap: %.4f') % (avg_equality_gap_0, avg_equality_gap_1, avg_parity_gap)
    print(s_eval)
    #print(eopp_list)
    #print(data_count)
    #eopp_list = eopp_list / data_count
    #max_eopp = torch.max(eopp_list, dim=0)[0] - torch.min(eopp_list, dim=0)[0]
    #max_eopp = torch.max(max_eopp)
    #print("\n",'max_eopps',max_eopp)
    return top1.avg, unfairness.avg, losses.avg, avg_equality_gap_0, avg_equality_gap_1, avg_parity_gap

def modifier(args, epoch, model):
    return
