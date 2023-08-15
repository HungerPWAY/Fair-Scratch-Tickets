import time
import torch
import tqdm
import torch.nn as nn
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
import torch.nn.functional as F

__all__ = ["train", "validate", "modifier"]


def floss_dp(outputs, sens_attr, args):
    Pa = args.Nmatrix[1,:].sum()/args.Nmatrix.sum()
    Pb = args.Nmatrix[0,:].sum()/args.Nmatrix.sum()
    #print(Pa,Pb)
    if args.fair_regularization == "logistic":
        
        return -args.lam_fair/outputs.shape[0] * (F.logsigmoid(outputs[sens_attr]).sum()/Pa + F.logsigmoid(-outputs[~sens_attr]).sum()/Pb)
    elif args.fair_regularization == "linear":
        return args.lam_fair/outputs.shape[0] * (-outputs[sens_attr].sum()/Pa + outputs[~sens_attr].sum()/Pb)
    elif args.fair_regularization == "hinge":
        baseline = torch.tensor(0.).cuda(args.gpu)
        return args.lam_fair/outputs.shape[0] * (torch.max(baseline,1-outputs[sens_attr]).sum()/Pa + torch.max(baseline,1+outputs[~sens_attr]).sum()/Pb)
    else:
        return 0

def floss_dp_2(outputs, sens_attr, args):
    Pa = args.Nmatrix[1,:].sum()/args.Nmatrix.sum()
    Pb = args.Nmatrix[0,:].sum()/args.Nmatrix.sum()
    #print(Pa,Pb)
    if args.fair_regularization == "logistic":
        
        return -args.lam_fair/outputs.shape[0] * (F.logsigmoid(outputs[~sens_attr]).sum()/Pb + F.logsigmoid(-outputs[sens_attr]).sum()/Pa)
    elif args.fair_regularization == "linear":
        return args.lam_fair/outputs.shape[0] * (-outputs[~sens_attr].sum()/args.Pb + outputs[sens_attr].sum()/args.Pa)
    elif args.fair_regularization == "hinge":
        baseline = torch.tensor(0.).cuda(args.gpu)
        return args.lam_fair/outputs.shape[0] * (torch.max(baseline,1-outputs[~sens_attr]).sum()/args.Pa + torch.max(baseline,1+outputs[sens_attr]).sum()/args.Pb)
    else:
        return 0


def floss_eoo(outputs, sens_attr, args):
    Pa = args.Nmatrix[1,1]/args.Nmatrix.sum()
    Pb = args.Nmatrix[0,1]/args.Nmatrix.sum()

    if args.fair_regularization == "logistic":
        return -args.lam_fair/outputs.shape[0] * (F.logsigmoid(outputs[sens_attr]).sum()/Pa + F.logsigmoid(-outputs[~sens_attr]).sum()/Pb)

    elif args.fair_regularization == "linear":
        return args.lam_fair/outputs.shape[0] * (-outputs[sens_attr].sum()/Pa + outputs[~sens_attr].sum()/Pb)

    elif args.fair_regularization == "hinge":
        baseline = torch.tensor(0.).cuda(args.gpu)
        return args.lam_fair/outputs.shape[0] * (torch.max(baseline,1-outputs[sens_attr]).sum()/Pa + torch.max(baseline,1+outputs[~sens_attr]).sum()/Pb)

    else:
        return 0


def floss_eoo_2(outputs, sens_attr, args):
    Pa = args.Nmatrix[1,1]/args.Nmatrix.sum()
    Pb = args.Nmatrix[0,1]/args.Nmatrix.sum()

    if args.fair_regularization == "logistic":
        return -args.lam_fair/outputs.shape[0] * (F.logsigmoid(outputs[sens_attr]).sum()/Pa + F.logsigmoid(-outputs[~sens_attr]).sum()/Pb)

    elif args.fair_regularization == "linear":
        return args.lam_fair/outputs.shape[0] * (-outputs[sens_attr].sum()/Pa + outputs[~sens_attr].sum()/Pb)

    elif args.fair_regularization == "hinge":
        baseline = torch.tensor(0.).cuda(args.gpu)
        return args.lam_fair/outputs.shape[0] * (torch.max(baseline,1-outputs[sens_attr]).sum()/Pa + torch.max(baseline,1+outputs[~sens_attr]).sum()/Pb)

    else:
        return 0



def train(train_loader, model, criterion, optimizer, epoch, args, writer , count = None):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    unfairness = AverageMeter("Unfairness", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, unfairness],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()
    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, targets, groups) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        if count is not None and count % 100 == 0:
            save = dict()
            save['state_dict'] = model.state_dict()
            torch.save(save, f"./save/save_{count}_{args.task}_{args.prune_rate}.pth")
        
        data_time.update(time.time() - end)
        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        groups = groups.cuda(args.gpu, non_blocking=True)



        # compute output
        output = model(images)


        if args.fair_type == "dp":
            sens_attr = groups.bool()

            if args.set == "CelebA" or args.set == "CelebA_B":

                loss = criterion(output.reshape(-1), targets.float()) + floss_dp(output.reshape(-1), sens_attr, args)
            elif args.set == "UTKFace2":

                loss = criterion(output.reshape(-1), targets.float()) + floss_dp_2(output.reshape(-1), sens_attr, args)
            else:

                loss = criterion(output.reshape(-1), targets.float()) + floss_dp_2(output.reshape(-1), sens_attr, args)

            preds = (output.reshape(-1) >= 0).float()


            cur_unfairness = torch.tensor([preds[sens_attr].sum(), preds[ sens_attr].shape[0],
                               preds[~sens_attr].sum(), preds[~sens_attr].shape[0]])

        elif args.fair_type == "eoo":
            sens_attr = groups.bool()
            labels_bool = targets.bool()
            if args.set == "CelebA" or "CelebA_B":
                loss = criterion(output.reshape(-1), targets.float()) + floss_eoo(output.reshape(-1)[labels_bool], sens_attr[labels_bool], args)
            elif args.set == "UTKFace2":
                loss = criterion(output.reshape(-1), targets.float()) + floss_eoo_2(output.reshape(-1)[labels_bool], sens_attr[labels_bool], args)
            else:
                loss = criterion(output.reshape(-1), targets.float()) + floss_eoo(output.reshape(-1)[labels_bool], sens_attr[labels_bool], args)

            preds = (output.reshape(-1) >= 0).float()

            cur_unfairness = torch.tensor([preds[ sens_attr & labels_bool].sum(), preds[ sens_attr & labels_bool].shape[0],
                              preds[~sens_attr & labels_bool].sum(), preds[~sens_attr & labels_bool].shape[0]])



        if args.set == "CelebA" or "CelebA_B":
            unfairness.update((cur_unfairness[2]/cur_unfairness[3] - cur_unfairness[0]/cur_unfairness[1]).item(), images.size(0))  
        elif args.set == "UTKFace2":
            if args.fair_type == "dp":
                unfairness.update((cur_unfairness[0]/cur_unfairness[1] - cur_unfairness[2]/cur_unfairness[3]).item(), images.size(0))
            elif args.fair_type =="eoo":
                unfairness.update((cur_unfairness[2]/cur_unfairness[3] - cur_unfairness[0]/cur_unfairness[1]).item(), images.size(0))
        else:
            unfairness.update((cur_unfairness[2]/cur_unfairness[3] - cur_unfairness[0]/cur_unfairness[1]).item(), images.size(0))
        
        acc1 = (targets == preds).float().mean()
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        #unfairness.update((cur_unfairness[2]/cur_unfairness[3] - cur_unfairness[0]/cur_unfairness[1]).item(), images.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if count is not None:
            count += 1            

        
        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)
            #print("unfairness", (running_unfair[2]/running_unfair[3] - running_unfair[0]/running_unfair[1]).item())


    return top1.avg, unfairness.avg, losses.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    unfairness = AverageMeter("Unfairness", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, unfairness], prefix="Val: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        #running_unfair = 0.0
        for i, (images, targets, groups) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):

            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
            groups = groups.cuda(args.gpu, non_blocking=True)



            # compute output
            output = model(images)

            sens_attr = groups.bool()
            loss = criterion(output.reshape(-1), targets.float()) + floss_dp(output.reshape(-1), sens_attr, args)

            preds = (output.reshape(-1) >= 0).float()

            #print(args.fair_type)
            if args.fair_type == "dp":
                sens_attr = groups.bool()
                loss = criterion(output.reshape(-1), targets.float()) + floss_dp(output.reshape(-1), sens_attr, args)

                preds = (output.reshape(-1) >= 0).float()


                cur_unfairness = torch.tensor([preds[sens_attr].sum(), preds[ sens_attr].shape[0],
                                preds[~sens_attr].sum(), preds[~sens_attr].shape[0]])
            elif args.fair_type == "eoo":
                sens_attr = groups.bool()
                labels_bool = targets.bool()
                if args.set == "CelebA" or "CelebA_B":
                    loss = criterion(output.reshape(-1), targets.float()) + floss_eoo(output.reshape(-1)[labels_bool], sens_attr[labels_bool], args)
                elif args.set == "UTKFace2":
                    loss = criterion(output.reshape(-1), targets.float()) + floss_eoo_2(output.reshape(-1)[labels_bool], sens_attr[labels_bool], args)


                preds = (output.reshape(-1) >= 0).float()

                cur_unfairness = torch.tensor([preds[ sens_attr & labels_bool].sum(), preds[ sens_attr & labels_bool].shape[0],
                              preds[~sens_attr & labels_bool].sum(), preds[~sens_attr & labels_bool].shape[0]])
            #print(cur_unfairness)

            acc1 = (targets == preds).float().mean()
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            if args.set == "CelebA" or "CelebA_B":
                unfairness.update((cur_unfairness[2]/cur_unfairness[3] - cur_unfairness[0]/cur_unfairness[1]).item(), images.size(0))  
            elif args.set == "UTKFace2":
                if args.fair_type == "dp":
                    unfairness.update((cur_unfairness[0]/cur_unfairness[1] - cur_unfairness[2]/cur_unfairness[3]).item(), images.size(0))
                elif args.fair_type =="eoo":
                    unfairness.update((cur_unfairness[2]/cur_unfairness[3] - cur_unfairness[0]/cur_unfairness[1]).item(), images.size(0))
            else:
                unfairness.update((cur_unfairness[2]/cur_unfairness[3] - cur_unfairness[0]/cur_unfairness[1]).item(), images.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_val_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="val", global_step=epoch)

    return top1.avg, unfairness.avg, losses.avg


class MMDLoss(nn.Module):
    def __init__(self, w_m, sigma, num_groups, num_classes, kernel):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, groups, labels):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
        else:
            student = f_s.view(f_s.shape[0], -1)

        mmd_loss = 0

        for c in range(self.num_classes):

            target_joint = student[labels == c].clone().detach()

            for g in range(self.num_groups):
                if len(student[(labels == c) * (groups == g)]) == 0:
                    continue

                K_SSg, sigma_avg = self.pdist(target_joint, student[(labels == c) * (groups == g)],
                                              sigma_base=self.sigma, kernel=self.kernel)

                K_SgSg, _ = self.pdist(student[(labels==c) * (groups==g)], student[(labels==c) * (groups==g)],
                                       sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                K_SS, _ = self.pdist(target_joint, target_joint,
                                     sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                mmd_loss += torch.clamp(K_SS.mean() + K_SgSg.mean() - 2 * K_SSg.mean(), 0.0, np.inf).mean()

        loss = self.w_m * mmd_loss / (2*self.num_groups)

        return loss

    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()

                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base**2)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)
        return res, sigma_avg


def modifier(args, epoch, model):
    return
