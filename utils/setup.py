# -*- coding: utf-8 -*-

import torch
from utils.dataloader import dataloader
import logging


def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        logging.info("CUDA GPU will be used for computations")
        device = torch.cuda.current_device()
    else:
        logging.info("CPU will be used for computations")
        device = torch.device('cpu')

    train_loader, test_loader = dataloader(args,
                                           args.dataset,
                                           None,
                                           None,
                                           batch_size=args.batch_size,
                                           val_batch_size=args.val_batch_size,
                                           workers=4)

    logging.info("Training set length: " + str(args.full_train_len))
    logging.info("Test set length: " + str(args.full_test_len))

    return device, train_loader, test_loader

