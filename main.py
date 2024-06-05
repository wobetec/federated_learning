#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import argparse

from src.experiments import BaseLine, Federated


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='baseline', help="experiment name")
    
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0, help='whether to use unequal data splits for  non-i.i.d setting (use 0 for equal splits)')

    # other arguments
    parser.add_argument('--gpu', default=None, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()

    if args.exp_name == 'baseline':
        exp = BaseLine(args)
        exp.train()
        print('Test on', len(exp.test_dataset), 'samples')
        print("Train Accuracy: {:.2f}%".format(100*exp.train_accuracy[-1]))
        print("Test Accuracy: {:.2f}%".format(100*exp.test_accuracy[-1]))

    elif args.exp_name == 'federated':
        exp = Federated(args)
        exp.run()   
    else:
        raise ValueError(f'Unrecognized experiment name: {args.exp_name}')
    

    