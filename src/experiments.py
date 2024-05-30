#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10


from pickletools import optimize
from tabnanny import verbose
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import perf_counter
import copy

import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

from src.models import MLPMnist, CNNMnist, CNNCifar
from src.sampling import MNIST, CIFAR
from src.fed import average_weights, LocalUpdate
from tensorboardX import SummaryWriter
import os


import matplotlib
import matplotlib.pyplot as plt

class Experiment:

    def __init__(self, args):
        self.exp_name = args.exp_name

        self.epochs = args.epochs
        self.num_users = args.num_users
        self.frac = args.frac
        self.local_ep = args.local_ep
        self.local_bs = args.local_bs
        self.lr = args.lr
        self.momentum = args.momentum

        self.model_name = args.model
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.num_channels = args.num_channels
        self.norm = args.norm
        self.num_filters = args.num_filters
        self.max_pool = args.max_pool

        self.dataset_name = args.dataset
        self.num_classes = args.num_classes
        self.gpu = args.gpu
        self.optimizer_name = args.optimizer
        self.iid = args.iid
        self.unequal = args.unequal
        self.stopping_rounds = args.stopping_rounds
        self.verbose = args.verbose
        self.seed = args.seed

        self.set_gpu(self.gpu)
        self.train_dataset, self.test_dataset = self.get_dataset()
        self.model = self.get_model()
        self.model.to(self.device)
        self.model.train()
        self.optimizer = self.set_optimizer()
        self.trainloader = self.get_dataloader()
        self.criterion = torch.nn.NLLLoss().to(self.device)

        self.logger = SummaryWriter(os.path.join(os.path.dirname(__file__), '../logs/'))

    def set_gpu(self, gpu):
        self.gpu = gpu
        if self.gpu:
            torch.cuda.set_device(self.gpu)
        self.device = 'cuda' if self.gpu else 'cpu'

    def get_model(self):
        if self.model_name == 'cnn':
            if self.dataset_name == 'mnist':
                return CNNMnist(self.num_channels, self.num_classes)
            elif self.dataset_name == 'cifar':
                return CNNCifar(self.num_classes)
        elif self.model_name == 'mlp':
            if self.dataset_name == 'mnist':
                img_shape = self.train_dataset[0][0].shape
                len_in = 1
                for x in img_shape:
                    len_in *= x
                return MLPMnist(dim_in=len_in, dim_hidden=200, dim_out=self.num_classes)
        else:
            raise ValueError(f'Unrecognized model name: {self.model_name}')
        
        raise ValueError(f'Unrecognized dataset name: {self.dataset_name}')
    
    def get_dataset(self):
        if self.dataset_name == 'mnist':
            return MNIST.load_dataset()
        elif self.dataset_name == 'cifar':
            return CIFAR.load_dataset()
        else:
            raise ValueError(f'Unrecognized dataset name: {self.dataset_name}')
        
    def get_sampling(self):
        if self.dataset_name == 'mnist':
            if self.iid:
                return MNIST.iid(self.num_users)
            elif not self.unequal:
                return MNIST.non_iid(self.num_users)
            else:
                return MNIST.non_iid_unequal(self.num_users)
        elif self.dataset_name == 'cifar':
            if self.iid:
                return CIFAR.iid(self.num_users)
            else:
                return CIFAR.non_iid(self.num_users)
        
        raise ValueError(f'Unrecognized sampling type')
    
    def set_optimizer(self):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        elif self.optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        else:
            raise ValueError(f'Unrecognized optimizer name: {self.optimizer_name}')
        
    def get_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)
    
    def test_inference(self):
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        criterion = nn.NLLLoss().to(self.device)
        testloader = DataLoader(self.test_dataset, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = self.model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        self.test_acc = accuracy
        self.test_loss = loss

    def details(self):
        print('\nExperimental details:')
        print(f'    Model     : {self.model_name}')
        print(f'    Optimizer : {self.optimizer_name}')
        print(f'    Learning  : {self.lr}')
        print(f'    Global Rounds   : {self.epochs}\n')

        print('    Federated parameters:')
        if self.iid:
            print('    IID')
        else:
            print('    Non-IID')
        print(f'    Fraction of users  : {self.frac}')
        print(f'    Local Batch size   : {self.local_bs}')
        print(f'    Local Epochs       : {self.local_ep}\n')
        
    def start_time(self):
        self.time_start = perf_counter()

    def end_time(self):
        self.time_end = perf_counter()

    def elapsed_time(self):
        self.time_elapsed = self.time_end - self.time_start
        

class BaseLine(Experiment):

    def __init__(self, args):
        super().__init__(args)
        
    def train(self):
        self.epoch_loss = []
        for epoch in tqdm(range(self.epochs)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 50 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(images), len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            loss_avg = sum(batch_loss)/len(batch_loss)
            print('\nTrain loss:', loss_avg)
            self.epoch_loss.append(loss_avg)
    
    def plot_loss(self):
        plt.figure()
        plt.plot(range(len(self.epoch_loss)), self.epoch_loss)
        plt.xlabel('epochs')
        plt.ylabel('Train loss')
        plt.savefig('./save/nn_{}_{}_{}.png'.format(self.dataset_name, self.model_name, self.epochs))
    
    def print_test(self):
        print('Test on', len(self.test_dataset), 'samples')
        print("Test Accuracy: {:.2f}%".format(100*self.test_acc))

    def run(self):
        self.details()
        self.train()
        self.plot_loss()
        self.test_inference()
        self.print_test()


class Federated(Experiment):

    def __init__(self, args):
        super().__init__(args)

        self.user_groups = self.get_sampling()

    def run(self):
        self.global_weights = self.model.state_dict()
        
        # Training
        train_loss, train_accuracy = [], []
        print_every = 2

        for epoch in tqdm(range(self.epochs)):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')

            self.model.train()
            m = max(int(self.frac * self.num_users), 1)
            idxs_users = np.random.choice(range(self.num_users), m, replace=False)

            for idx in idxs_users:
                local_model = LocalUpdate(
                    dataset=self.train_dataset,
                    idxs=self.user_groups[idx],
                    logger=self.logger,
                    optimizer=self.optimizer_name,
                    lr=self.lr,
                    local_ep=self.local_ep,
                    local_bs=self.local_bs,
                    verbose=self.verbose,
                    device=self.device
                )
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(self.model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            self.model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.model.eval()
            for c in range(self.num_users):
                local_model = LocalUpdate(
                    dataset=self.train_dataset,
                    idxs=self.user_groups[idx],
                    logger=self.logger,
                    optimizer=self.optimizer_name,
                    lr=self.lr,
                    local_ep=self.local_ep,
                    verbose=self.verbose,
                    device=self.device
                )
                acc, loss = local_model.inference(model=self.model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
        
        matplotlib.use('Agg')

        plt.figure()
        plt.title('Training Loss vs Communication rounds')
        plt.plot(range(len(train_loss)), train_loss, color='r')
        plt.ylabel('Training loss')
        plt.xlabel('Communication Rounds')
        plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                    format(self.dataset_name, self.model_name, self.epochs, self.frac,
                        self.iid, self.local_ep, self.local_bs))
        
        # Plot Average Accuracy vs Communication rounds
        plt.figure()
        plt.title('Average Accuracy vs Communication rounds')
        plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
        plt.ylabel('Average Accuracy')
        plt.xlabel('Communication Rounds')
        plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                    format(self.dataset_name, self.model_name, self.epochs, self.frac,
                        self.iid, self.local_ep, self.local_bs))