# -*- coding: utf-8 -*-

import argparse
from utils import setup, train
import logging
import os
import models.layers.surrogate_gradients as gradients


def main():
    parser = argparse.ArgumentParser(description='S-TLLR implementation')
    # General
    parser.add_argument('--arch', type=str, default='dvs_vgg_stllr',
                        help='SNN architecture.')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Disable CUDA training and run training on CPU')
    parser.add_argument('--dataset', type=str, default='DVSGesture',
                        help='Choice of the dataset')
    parser.add_argument('--save-path', type=str, default='./experiments/default',
                        help='Directory to save the checkpoint and logs of the experiment')
    parser.add_argument('--data-path', type=str,
                        help='Path for the datasets folder. The datasets is going to be downloaded if it is not in the location.')
    parser.add_argument('--trials', type=int, default=1,
                        help='Number of trial experiments to do (i.e. repetitions with different initializations)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'NAG', 'Adam', 'RMSProp', 'RProp'], default='Adam',
                        help='Choice of the optimizer')
    parser.add_argument('--loss', type=str, choices=['MSE', 'BCE', 'CE'], default='CE',
                        help='Choice of the loss function')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=5,
                        help='Batch size for testing')
    parser.add_argument('--label-encoding', type=str, default="class", choices=["class", "one-hot"],
                        help='Label encoding by default class. But, one-hot should be use for DFA.')
    parser.add_argument('--activation', type=str, default='LinearSpike', choices=gradients.__dict__["__all__"],
                        help='Name of the secondary activation function (Psi).')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Seed for reproducibility.')
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path for the pretrained model')
    parser.add_argument('--training-mode', type=str, default='stllr', choices=["stllr_online", "stllr", "bptt"],
                        help='Training mode.')
    parser.add_argument('--feedback-mode', type=str, default='BP', choices=["BP", "DFA", "sDFA", "SBP", "LocalLoss"],
                        help='Feedback mode to produce the learning signal.')
    parser.add_argument('--delay-ls', type=int, default=5,
                        help='Number of time steps for which the learning signal is available (T - T_l).')
    parser.add_argument('--scheduler', type=int, default=0,
                        help='Learning rate decay time.')
    parser.add_argument('--print-freq', type=int, default=200,
                        help='Frequency of printing results.')
    parser.add_argument('--factors-stdp', nargs='+', type=float, default=[0.2, 0.75, -1, 1],
                        help='STDP parameters $[lambda_{post}, lambda_{pre}, alpha_{post}, alpha_{pre}]$.')
    parser.add_argument('--pooling', type=str, default='MAX', choices=["MAX", "AVG"],
                        help='Pooling layer.')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Weight decay L2 normalization')

    args = parser.parse_args()

    # Create a new folder in 'args.save_path' to save the results of the experiment
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Log configuration
    log_path = args.save_path + "/log.log"
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=log_path)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)
    logging.info('=> Everything will be saved to {}'.format(args.save_path))

    # Initiate the training
    device, train_loader, test_loader = setup.setup(args)
    train.train(args, device, train_loader, test_loader)


if __name__ == '__main__':
    main()