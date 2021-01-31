#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
from pathlib import Path
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import Mfirrn
import torch.backends.cudnn as cudnn

from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz
from utils.ddfa import str2bool, AverageMeter
from utils.io import mkdir
from vdc_loss import VDCLoss
from wpdc_loss import WPDCLoss
import random

# global args (configuration)
args = None
lr = None
# arch_choices = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']
arch_choices = ['resnet34']


def parse_args():
    parser = argparse.ArgumentParser(description='3DMM Fitting')
    parser.add_argument('-j', '--workers', default=6, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-vb', '--val-batch-size', default=32, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--print-freq', '-p', default=20, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--devices-id', default='0', type=str)
    parser.add_argument('--filelists-train',
                        default='', type=str)
    parser.add_argument('--filelists-val',
                        default='', type=str)
    parser.add_argument('--root', default='')
    parser.add_argument('--snapshot', default='', type=str)
    parser.add_argument('--log-file', default='output.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--size-average', default='true', type=str2bool)
    parser.add_argument('--num-classes', default=62, type=int)
    parser.add_argument('--arch', default='resnet34', type=str,
                        choices=arch_choices)  # ************************************************MobileNetV3_Large
    # parser.add_argument('--arch', default='MobileNetV3', type=str,choices=arch_choices)
    parser.add_argument('--frozen', default='false', type=str2bool)
    parser.add_argument('--milestones', default='30,40,50', type=str)
    parser.add_argument('--task', default='all', type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)
    parser.add_argument('--warmup', default=-1, type=int)
    parser.add_argument('--param-fp-train',
                        default='',
                        type=str)
    parser.add_argument('--param-fp-val',
                        default='')
    parser.add_argument('--opt-style', default='resample', type=str)  # resample
    parser.add_argument('--resample-num', default=132, type=int)
    parser.add_argument('--loss', default='vdc', type=str)
    global args
    args = parser.parse_args()

    # some other operations
    args.devices_id = [int(d) for d in args.devices_id.split(',')]
    args.milestones = [int(m) for m in args.milestones.split(',')]

    snapshot_dir = osp.split(args.snapshot)[0]
    mkdir(snapshot_dir)


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    global lr
    lr = args.base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()

    model.train()

    end = time.time()
    # loader is batch style
    # for i, (input, target) in enumerate(train_loader):
    for i, (input, target) in enumerate(train_loader):
        target.requires_grad = False
        target = target.cuda(non_blocking=True)

        input1 = jigsaw_generator(input, 2)
        # input2 = jigsaw_generator(input, 3)
        # output, output_medium, output_fine, concat_out = model(input, input1, input2)
        output, output_fine, concat_out = model(input, input )

        # if args.loss.lower() == 'vdc':
        #     loss1 = criterion(output_medium, target)
        # elif args.loss.lower() == 'wpdc':
        #     loss1 = criterion(output_medium, target)
        #     # print(loss)
        # elif args.loss.lower() == 'pdc':
        #
        #     loss1 = criterion(output_medium, target)
        # else:
        #     raise Exception(f'Unknown loss {args.loss}')

        if args.loss.lower() == 'vdc':
            loss2 = criterion(output_fine, target)
        elif args.loss.lower() == 'wpdc':
            loss2 = criterion(output_fine, target)
        elif args.loss.lower() == 'pdc':
            loss2 = criterion(output_fine, target)
        else:
            raise Exception(f'Unknown loss {args.loss}')

        if args.loss.lower() == 'vdc':
            loss = criterion(output, target)
        elif args.loss.lower() == 'wpdc':
            loss = criterion(output, target)
        elif args.loss.lower() == 'pdc':
            loss = criterion(output, target)
        else:
            raise Exception(f'Unknown loss {args.loss}')

        if args.loss.lower() == 'vdc':
            loss3 = criterion(concat_out, target)
        elif args.loss.lower() == 'wpdc':
            loss3 = criterion(concat_out, target)
        elif args.loss.lower() == 'pdc':
            loss3 = criterion(concat_out, target)
        else:
            raise Exception(f'Unknown loss {args.loss}')

        optimizer.zero_grad()
        # loss_all = loss + loss1 + loss2 + loss3
        loss_all = loss + loss2 + loss3
        loss_all.backward()
        optimizer.step()

        data_time.update(time.time() - end)

        losses.update(loss.item(), input.size(0))
        losses1.update(loss2.item(), input.size(0))
        losses2.update(loss2.item(), input.size(0))
        losses3.update(loss3.item(), input.size(0))
        # compute gradient and do SGD step

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # log
        if i % args.print_freq == 0:
            logging.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                         f'LR: {lr:8f}\t'
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         f'Loss_coarse {losses.val:.4f} ({losses.avg:.4f})\t'
                         # f'Loss_medium {losses1.val:.4f} ({losses1.avg:.4f})\t'
                         f'Loss_fine {losses2.val:.4f} ({losses2.avg:.4f})\t'
                         f'Loss_across_all {losses3.val:.4f} ({losses3.avg:.4f}\t)')


def jigsaw_generator(inputs, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 120 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = inputs.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        # print(temp.size())
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                   y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


def validate(val_loader, model, criterion, epoch):
    model.eval()

    end = time.time()
    with torch.no_grad():
        losses = []
        losses1 = []
        losses2 = []
        losses3 = []
        losses4 = []
        for i, (input, target) in enumerate(val_loader):
            # compute output
            target.requires_grad = False
            target = target.cuda(non_blocking=True)
            output = model(input, 4)
            input1 = jigsaw_generator(input, 4)
            input2 = jigsaw_generator(input, 2)

            output1 = model(input1, 1)
            output2 = model(input2, 2)
            output3 = model(torch.cat((output, output1, output2), dim=-1), 3)

            attention_tem = torch.cat((output.unsqueeze(dim=-2), output1.unsqueeze(dim=-2)), dim=-2)
            attention_tem_dog = torch.cat((attention_tem, output2.unsqueeze(dim=-2)), dim=-2)
            attention_input = torch.cat((attention_tem_dog, output3.unsqueeze(dim=-2)), dim=-2)

            output4 = model(attention_input, 5)
            out_ori = output4[:, 0, :]
            out_head1 = output4[:, 1, :]
            out_head2 = output4[:, 2, :]
            out_head3 = output4[:, 3, :]
            tem = torch.add(out_ori, out_head1)
            attention_out = torch.add(tem, out_head2)
            attention_out = torch.add(attention_out, out_head3)
            attention_out = attention_out.squeeze(1)

            loss = criterion(output, target)
            loss1 = criterion(output1, target)
            loss2 = criterion(output2, target)
            loss3 = criterion(output3, target)
            loss4 = criterion(attention_out, target)

            losses.append(loss.item())
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            losses4.append(loss4.item())

        elapse = time.time() - end
        # loss = torch.mean(torch.stack(losses))   #*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*×*××*×
        loss = np.mean(losses)
        loss1 = np.mean(losses1)
        loss2 = np.mean(losses2)
        loss3 = np.mean(losses3)
        loss4 = np.mean(losses4)
        logging.info(f'Val: [{epoch}][{len(val_loader)}]\t'
                     f'Loss {loss:.4f}\t'
                     f'Loss_coarse {loss:.4f}\t'
                     f'Loss_fine {loss1:.4f}\t'
                     f'Loss_medium {loss2:.4f}\t'
                     f'Loss_across_all {loss3:.4f}\t)'
                     f'Loss_attention {loss4:.4f}\t)'
                     f'Time {elapse:.3f}')


def main():
    parse_args()  # parse global argsl

    # logging setup
    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode=args.log_mode),
            logging.StreamHandler()
        ]
    )

    print_args(args)  # print args

    # step1: define the model structure
    model = getattr(Mfirrn, args.arch)(num_classes=62)
    nparameters = sum(p.numel() for p in model.parameters())
    # print(model)
    print('Total number of parameters: %d' % nparameters)
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    torch.cuda.set_device(args.devices_id[0])  # fix bug for `ERROR: all tensors must be on devices[0]`

    model = nn.DataParallel(model, device_ids=args.devices_id).cuda()  # -> GPU
    # step2: optimization: loss and optimization method
    # criterion = nn.MSELoss(size_average=args.size_average).cuda()
    if args.loss.lower() == 'wpdc':
        print(args.opt_style)
        criterion = WPDCLoss(opt_style=args.opt_style).cuda()
        logging.info('Use WPDC Loss')
    elif args.loss.lower() == 'vdc':
        criterion = VDCLoss(opt_style=args.opt_style).cuda()
        logging.info('Use VDC Loss')
    elif args.loss.lower() == 'pdc':
        criterion = nn.MSELoss(size_average=args.size_average).cuda()
        logging.info('Use PDC loss')
    else:
        raise Exception(f'Unknown Loss {args.loss}')

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # step 2.1 resume
    if args.resume:
        if Path(args.resume).is_file():
            logging.info(f'=> loading checkpoint {args.resume}')

            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
            # checkpoint = torch.load(args.resume)['state_dict']
            model.load_state_dict(checkpoint)

        else:
            logging.info(f'=> no checkpoint found at {args.resume}')

    # step3: data
    normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

    train_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )
    val_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_val,
        param_fp=args.param_fp_val,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=True)

    # step4: run
    cudnn.benchmark = True
    if args.test_initial:
        logging.info('Testing from initial')
        validate(val_loader, model, criterion, args.start_epoch)

    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, args.milestones)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        filename = f'{args.snapshot}_checkpoint_epoch_{epoch}.pth.tar'
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict()
            },
            filename
        )

        # validate(val_loader, model, criterion, epoch)


if __name__ == '__main__':
    main()
