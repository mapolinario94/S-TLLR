# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import models
import logging
import time
from utils.metrics import AverageMeter, ProgressMeter, accuracy
import random


def train(args, device, train_loader, test_loader):
    if args.seed != 0:
        set_seed(args.seed)

    for trial in range(1, args.trials + 1):

        # Network topology
        model = models.__dict__[args.arch](args, device)
        if trial == 1:
            logging.info(f'Total Parameters: {int(10*(sum(p.numel() for p in model.parameters()) / 1000000.0))/10}M')
        # Use CUDA for GPU-based computation if enabled
        if args.cuda:
            model.cuda()

        # Initial monitoring
        if (args.trials > 1):
            logging.info('\nIn trial {} of {}'.format(trial, args.trials))
        if (trial == 1):
            logging.info("=== Model ===")
            logging.info(model)

        # Optimizer
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        elif args.optimizer == 'RProp':
            optimizer = optim.Rprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")

        if args.scheduler > 0:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.scheduler)
        else:
            scheduler = None

        if args.loss == 'MSE':
            loss = nn.MSELoss()
        elif args.loss == 'BCE':
            loss = nn.BCELoss()
        elif args.loss == 'CE':
            loss = nn.CrossEntropyLoss()
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")

        # Training and performance monitoring
        logging.info("\n=== Starting model training with %d epochs:\n" % (args.epochs,))
        best_acc1 = 0
        acc_train_hist = []
        acc_val_hist = []
        loss_train_hist = []
        loss_val_hist = []
        for epoch in range(1, args.epochs + 1):
            logging.info("\t Epoch " + str(epoch) + "...")
            # Will display the average accuracy on the training set during the epoch (changing weights)
            acc_t, loss_t = do_epoch(args, True, model, device, train_loader, optimizer, loss, 'train', epoch)
            acc_train_hist.append(acc_t.cpu().numpy())
            loss_train_hist.append(loss_t)
            # Check performance on the training set and on the test set:
            if not args.skip_test:
                acc1, loss_val = do_epoch(args, False, model, device, test_loader, optimizer, loss, 'test', epoch)
                acc_val_hist.append(acc1.cpu().numpy())
                loss_val_hist.append(loss_val)
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                logging.info(f'Best acc at epoch {epoch}: {best_acc1}')
                if is_best:
                    if is_best:
                        state = {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(state, args.save_path + f'/trial_{trial}_model_best.pth.tar')
            if scheduler:
                scheduler.step()
        np.save(args.save_path + f'/trial_{trial}_train_acc.npy', np.array(acc_train_hist))
        np.save(args.save_path + f'/trial_{trial}_val_acc.npy', np.array(acc_val_hist))
        np.save(args.save_path + f'/trial_{trial}_train_loss.npy', np.array(loss_train_hist))
        np.save(args.save_path + f'/trial_{trial}_val_loss.npy', np.array(loss_val_hist))


def do_epoch(args, do_training: bool, model, device, loader, optimizer, loss_fct, benchType, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if benchType == 'train':
        progress = ProgressMeter(
            len(loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
    else:
        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

    if not do_training:
        model.eval()
    else:
        model.train()
    score = 0
    loss = 0
    batch = args.batch_size if (benchType == 'train') else args.val_batch_size
    length = args.full_train_len if (benchType == 'train') else args.full_test_len

    end = time.time()
    for batch_idx, (data, label) in enumerate(loader):
        data_time.update(time.time() - end)

        label = label.type(torch.int64)
        data, label = data.float().to(device), label.to(device)

        data, label, target, timesteps = data_resizing(args, data, label, device)

        args.n_steps = timesteps

        model.reset_states()
        if not do_training:
            with torch.no_grad():
                pred = 0
                for t in range(args.n_steps):
                    input = data[t] if data.size(0) > 1 else data[0]
                    output = model(input, None)
                    pred += output
        elif args.training_mode == "stllr_online":
            pred = 0
            for t in range(args.n_steps):
                input = data[t] if data.size(0) > 1 else data[0]
                output = model(input, target)
                pred += output
                if (data.size(0) - t) <= args.delay_ls:
                    optimizer.zero_grad()
                    loss = loss_fct(output, label)
                    loss.backward()
                    optimizer.step()
        elif args.training_mode == "bptt":
            optimizer.zero_grad()
            pred = 0
            for t in range(args.n_steps):
                input = data[t] if data.size(0) > 1 else data[0]
                pred += model(input, target)
            loss = loss_fct(pred, label)
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            pred = 0
            for t in range(args.n_steps):
                input = data[t] if data.size(0) > 1 else data[0]
                output = model(input, target)
                pred += output.detach()
                if (data.size(0) - t) <= args.delay_ls:
                    loss = loss_fct(output, label)
                    loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss = loss_fct(pred, label)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(pred, label, topk=(1, 5))
        losses.update(loss.item(), data.size(1))
        top1.update(acc1[0], data.size(1))
        top5.update(acc5[0], data.size(1))

        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.print_freq == (args.print_freq-1):
            progress.display(batch_idx)

    if benchType == 'train':
        logging.info(' @Training * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    else:
        logging.info(' @Testing * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg, losses.avg


def data_resizing(args, data, label, device):
    timesteps = data.size(1)
    batch_size = data.size(0)
    if args.dataset == 'DVSGesture':
        data = data.view(batch_size, timesteps, 2, 32, 32)
        data = data.permute(1, 0, 2, 3, 4)
        # label = label.unsqueeze(1).expand(batch_size, timesteps)
    elif args.dataset == 'NCALTECH101':
        data = data.view(batch_size, timesteps, 2, 45, 60)
        data = data.permute(1, 0, 2, 3, 4)
    elif args.dataset == 'DVSGesture100':
        data = data.view(batch_size, timesteps, 2, 64, 64)
        data = data.permute(1, 0, 2, 3, 4)
        # label = label.unsqueeze(1).expand(batch_size, timesteps)
    elif args.dataset == 'SHD':
        data = data.view(batch_size, timesteps, 700)
        data = data.permute(1, 0, 2)
        # timesteps = args.n_steps
        # label = label.unsqueeze(1).expand(batch_size, timesteps)
    elif args.dataset == 'MNIST':
        data = data.view(batch_size, 1, 1, 28, 28)
        data = data.permute(1, 0, 2, 3, 4)
        timesteps = args.n_steps
    elif args.dataset == 'RowMNIST':
        data = data.view(batch_size, 28, 28)
        data = data.permute(1, 0, 2)
        timesteps = args.n_steps
        # label = label.unsqueeze(1).expand(batch_size, 28)
    elif args.dataset == 'SMNIST':
        data = data.view(batch_size, -1, 1)
        data = data.permute(1, 0, 2)
        timesteps = args.n_steps
    elif args.dataset == 'PMNIST':
        data = data.view(batch_size, -1, 1)
        data = data.permute(1, 0, 2)
        timesteps = args.n_steps
    elif args.dataset == 'NMNIST':
        data = data.view(batch_size, timesteps, 2, 34, 34)
        data = data.permute(1, 0, 2, 3, 4)
        # label = label.unsqueeze(1).expand(batch_size, timesteps)
    elif args.dataset == 'CIFAR10DVS':
        data = data.view(batch_size, timesteps, 2, 48, 48)
        data = data.permute(1, 0, 2, 3, 4)
        # label = label.unsqueeze(1).expand(batch_size, timesteps)
    elif args.dataset == 'CIFAR10':
        data = data.view(batch_size, 1, 3, 32, 32)
        data = data.permute(1, 0, 2, 3, 4)
        timesteps = args.n_steps
    elif args.dataset == 'CIFAR100':
        data = data.view(batch_size, 1, 3, 32, 32)
        data = data.permute(1, 0, 2, 3, 4)
        timesteps = args.n_steps
    elif args.dataset == 'IMAGENET':
        data = data.view(batch_size, 1, 3, 224, 224)
        data = data.permute(1, 0, 2, 3, 4)
        timesteps = args.n_steps
    else:
        logging.info("ERROR: {0} is not supported".format(args.dataset))
        raise NameError("{0} is not supported".format(args.dataset))

    if args.classif and args.label_encoding == "one-hot":  # Do a one-hot encoding for classification
        target = F.one_hot(label, num_classes=args.n_classes).float()

    else:
        target = label
    label = label.view(-1, )

    return data, label, target, timesteps


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True