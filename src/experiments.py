#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import copy
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.sampling import MNIST, CIFAR
from src.fed import average_weights, LocalUpdate
from src.models import MLPMnist, CNNMnist, CNNCifar


class Experiment:

    def __init__(self, args):
        # Model parameters
        self.exp_name = args.exp_name

        self.epochs = args.epochs
        try:
            self.num_users = args.num_users
            self.frac = args.frac
            self.local_ep = args.local_ep
            self.local_bs = args.local_bs
        except:
            pass
        self._lr = args.lr
        self.lr = args.lr

        self.model_name = args.model
        self.optimizer_name = args.optimizer

        self.dataset_name = args.dataset
        self.num_classes = args.num_classes
        self.iid = args.iid
        self.unequal = args.unequal

        self.gpu = args.gpu
        self.verbose = args.verbose

        # Model data
        self.train_accuracy = []
        self.train_loss = []
        self.test_accuracy = []
        self.test_loss = []

        self.set_gpu(self.gpu)
        self.train_dataset, self.test_dataset = self.get_dataset()
        self.model = self.get_model()
        self.model.to(self.device)
        self.model.train()
        self.optimizer = self.set_optimizer()
        self.trainloader = self.get_dataloader()
        self.criterion = torch.nn.NLLLoss().to(self.device)

    def set_gpu(self, gpu):
        self.gpu = gpu
        if self.gpu:
            torch.cuda.set_device(self.gpu)
        self.device = 'cuda' if self.gpu else 'cpu'

    def get_model(self):
        if self.model_name == 'cnn':
            if self.dataset_name == 'mnist':
                return CNNMnist(self.num_classes)
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
        self.model.train()
        accuracy = correct/total

        return accuracy, loss

    def to_dict(self):
        if self.exp_name == 'baseline':
            return {
                'exp_name': self.exp_name,

                'epochs': self.epochs,
                'lr': self._lr,

                'model_name': self.model_name,
                'optimizer_name': self.optimizer_name,

                'dataset_name': self.dataset_name,
                'num_classes': self.num_classes,
                'iid': self.iid,
                'unequal': self.unequal,

                'train_accuracy': self.train_accuracy,
                'train_loss': self.train_loss,
                'test_accuracy': self.test_accuracy,
                'test_loss': self.test_loss,
            }
        return {
            'exp_name': self.exp_name,

            'epochs': self.epochs,
            'num_users': self.num_users,
            'frac': self.frac,
            'local_ep': self.local_ep,
            'local_bs': self.local_bs,
            'lr': self._lr,

            'model_name': self.model_name,
            'optimizer_name': self.optimizer_name,

            'dataset_name': self.dataset_name,
            'num_classes': self.num_classes,
            'iid': self.iid,
            'unequal': self.unequal,

            'train_accuracy': self.train_accuracy,
            'train_loss': self.train_loss,
            'test_accuracy': self.test_accuracy,
            'test_loss': self.test_loss,
        }


class BaseLine(Experiment):

    def __init__(self, args):
        super().__init__(args)
        
    def train(self):
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
                    print(f'Train Epoch: {epoch+1} [{batch_idx * len(images)}/{len(self.trainloader.dataset)} ({100. * batch_idx / len(self.trainloader):.0f}%)]\tLoss: {loss.item():.6f}')
                batch_loss.append(loss.item())
            loss_avg = sum(batch_loss)/len(batch_loss)
            print('\nTrain loss:', loss_avg)

            self.train_loss.append(loss_avg)

            # Calculate Train accuracy
            self.model.eval()
            total, correct = 0.0, 0.0
            trainloader = DataLoader(self.train_dataset, batch_size=128,
                                    shuffle=False)
            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = self.model(images)

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
            self.model.train()

            train_accuracy = correct/total
            self.train_accuracy.append(train_accuracy)

            test_accuracy, test_loss = self.test_inference()
            self.test_accuracy.append(test_accuracy)
            self.test_loss.append(test_loss)

            if self.model_name == 'cnn':
                self.lr = self.lr * 0.98

            if self.test_accuracy[-1] >= 0.99:
                break


class Federated(Experiment):

    def __init__(self, args):
        super().__init__(args)

        self.user_groups = self.get_sampling()

    def train(self):
        self.global_weights = self.model.state_dict()
        
        # Training
        print_every = 2

        for epoch in tqdm(range(self.epochs)):
            local_weights, local_losses = [], []
            if self.verbose:
                print(f'\n | Global Training Round : {epoch+1} |\n')

            self.model.train()
            m = max(int(self.frac * self.num_users), 1)
            if m == self.num_users:
                idxs_users = range(self.num_users)
            else:
                idxs_users = np.random.choice(range(self.num_users), m, replace=False)

            for idx in idxs_users:
                local_model = LocalUpdate(
                    dataset=self.train_dataset,
                    idxs=self.user_groups[idx],
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
            self.train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.model.eval()
            for c in range(self.num_users):
                local_model = LocalUpdate(
                    dataset=self.train_dataset,
                    idxs=self.user_groups[idx],
                    optimizer=self.optimizer_name,
                    lr=self.lr,
                    local_ep=self.local_ep,
                    local_bs=self.local_bs,
                    verbose=self.verbose,
                    device=self.device
                )
                acc, loss = local_model.inference(model=self.model)
                list_acc.append(acc)
                list_loss.append(loss)
            self.train_accuracy.append(sum(list_acc)/len(list_acc))

            test_accuracy, test_loss = self.test_inference()
            self.test_accuracy.append(test_accuracy)
            self.test_loss.append(test_loss)

            if self.model_name == 'cnn':
                self.lr = self.lr * 0.98

            # print global training loss after every 'i' rounds
            if self.verbose:
                if (epoch+1) % print_every == 0:
                    print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                    print(f'Training Loss : {np.mean(np.array(self.train_loss))}')
                    print(f'Train Accuracy: {100*self.train_accuracy[-1]:.2f}% \n')

            if self.test_accuracy[-1] >= 0.99:
                break
