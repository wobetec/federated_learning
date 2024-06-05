#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import os
import numpy as np
from torchvision import datasets, transforms
import requests


class Sampling:
    """
    Abstract class for sampling methods
    """

    @staticmethod
    def _download_dataset(dataset_name: str) -> tuple:
        """
        Download the dataset and apply the necessary transformations

        Params:
            dataset_name (str): Name of the dataset to download
            
        Returns
            train_dataset, test_dataset: Tuple containing the training and test datasets
        """

        data_folder = os.path.join(os.path.dirname(__file__), '../data/')
        if not os.path.exists(data_folder):
            os.makedirs(os.path.join(data_folder, 'cifar'))
            os.makedirs(os.path.join(data_folder, 'mnist'))
            os.makedirs(os.path.join(data_folder, 'cwws'))

        if dataset_name == 'cifar':
            # Download CIFAR-10 dataset from torchvision
            data_dir = os.path.join(data_folder, 'cifar/')
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                             transform=apply_transform)

            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=apply_transform)
    
        elif dataset_name == 'mnist':
            # Download MNIST dataset from torchvision
            data_dir = os.path.join(data_folder, 'mnist/')
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

        elif dataset_name == 'cwws':
            data_dir = os.path.join(data_folder, 'cwws/')
            file_path = os.path.join(data_dir, "shakespeare.txt")
            
            if not os.path.exists(file_path):
                # Download the dataset
                url = "https://www.gutenberg.org/files/100/100-0.txt"
                response = requests.get(url)
                os.makedirs(data_dir, exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(response.content)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.readlines()

            lines = [line.strip() for line in text if line.strip()]
            train_size = int(0.8 * len(lines))
            
            train_data = lines[:train_size]
            test_data = lines[train_size:]

            return train_data, test_data
            
        else:
            raise ValueError('Dataset not recognized')
        
        return train_dataset, test_dataset
    

class MNIST(Sampling):
    """
    Wrapper class for MNIST dataset
    """

    @classmethod
    def load_dataset(cls):
        return cls._download_dataset('mnist')

    @classmethod
    def iid(cls, num_users: int) -> dict:
        """
        Sample I.I.D. client data from MNIST dataset

        Params:
            num_users (int): Number of clients

        Returns:
            dict_users (dict): Dictionary containing the indices of the samples
            assigned to each client
        """

        train_dataset, _ = cls.load_dataset()
        num_items = int(len(train_dataset)/num_users)
        dict_users, all_idxs = {}, [i for i in range(len(train_dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    @classmethod
    def non_iid(cls, num_users: int) -> dict:
        """
        Sample non-I.I.D. client data from MNIST dataset

        Params:
            num_users (int): Number of clients

        Returns:
            dict_users (dict): Dictionary containing the indices of the samples
            assigned to each client
        """

        train_dataset, _ = cls.load_dataset()
        # 60,000 training imgs -->  200 imgs/shard X 300 shards
        num_shards, num_imgs = 200, 300
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(num_users)}
        idxs = np.arange(num_shards*num_imgs)
        labels = train_dataset.train_labels.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign 2 shards/client
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        return dict_users
    
    @classmethod
    def non_iid_unequal(cls, num_users: int) -> dict:
        """
        Sample non-I.I.D. client data from MNIST dataset with unequal splits

        Params:
            num_users (int): Number of clients

        Returns:
            dict_users (dict): Dictionary containing the indices of the samples
            assigned to each client
        """

        train_dataset, _ = cls.load_dataset()
        # 60,000 training imgs --> 50 imgs/shard X 1200 shards
        num_shards, num_imgs = 1200, 50
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(num_users)}
        idxs = np.arange(num_shards*num_imgs)
        labels = train_dataset.train_labels.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # Minimum and maximum shards assigned per client:
        min_shard = 1
        max_shard = 30

        # Divide the shards into random chunks for every client
        # s.t the sum of these chunks = num_shards
        random_shard_size = np.random.randint(min_shard, max_shard+1,
                                            size=num_users)
        random_shard_size = np.around(random_shard_size /
                                    sum(random_shard_size) * num_shards)
        random_shard_size = random_shard_size.astype(int)

        # Assign the shards randomly to each client
        if sum(random_shard_size) > num_shards:

            for i in range(num_users):
                # First assign each client 1 shard to ensure every client has
                # atleast one shard of data
                rand_set = set(np.random.choice(idx_shard, 1, replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)

            random_shard_size = random_shard_size-1

            # Next, randomly assign the remaining shards
            for i in range(num_users):
                if len(idx_shard) == 0:
                    continue
                shard_size = random_shard_size[i]
                if shard_size > len(idx_shard):
                    shard_size = len(idx_shard)
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                                replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)
        else:

            for i in range(num_users):
                shard_size = random_shard_size[i]
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                                replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)

            if len(idx_shard) > 0:
                # Add the leftover shards to the client with minimum images:
                shard_size = len(idx_shard)
                # Add the remaining shard to the client with lowest data
                k = min(dict_users, key=lambda x: len(dict_users.get(x)))
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                                replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[k] = np.concatenate(
                        (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)

        return dict_users



class CIFAR(Sampling):
    """
    Wrapper class for CIFAR-10 dataset
    """

    @classmethod
    def load_dataset(cls):
        return cls._download_dataset('cifar')

    @classmethod
    def iid(cls, num_users: int) -> dict:
        """
        Sample I.I.D. client data from CIFAR-10 dataset

        Params:
            num_users (int): Number of clients

        Returns:
            dict_users (dict): Dictionary containing the indices of the samples
            assigned to each client
        """
        train_dataset, _ = cls.load_dataset()
        num_items = int(len(train_dataset)/num_users)
        dict_users, all_idxs = {}, [i for i in range(len(train_dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    @classmethod
    def non_iid(cls, num_users: int) -> dict:
        """
        Sample non-I.I.D. client data from CIFAR-10 dataset

        Params:
            num_users (int): Number of clients

        Returns:
            dict_users (dict): Dictionary containing the indices of the samples
            assigned to each client
        """
        train_dataset, _ = cls.load_dataset()
        num_shards, num_imgs = 200, 250
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(num_users)}
        idxs = np.arange(num_shards*num_imgs)
        # labels = dataset.train_labels.numpy()
        labels = np.array(train_dataset.train_labels)

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        return dict_users
        pass


class CWWS(Sampling):
    """
    Wrapper class for CWWS dataset
    """

    @classmethod
    def load_dataset(cls):
        return cls._download_dataset('cwws')

    @classmethod
    def iid(cls, num_users: int) -> dict:
        """
        Sample I.I.D. client data from CWWS dataset

        Params:
            num_users (int): Number of clients

        Returns:
            dict_users (dict): Dictionary containing the indices of the samples
            assigned to each client
        """
        train_data, _ = cls.load_dataset()
        num_items = int(len(train_data) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(train_data))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    @classmethod
    def non_iid(cls, num_users: int) -> dict:
        """
        Sample non-I.I.D. client data from CWWS dataset

        Params:
            num_users (int): Number of clients

        Returns:
            dict_users (dict): Dictionary containing the indices of the samples
            assigned to each client
        """
        train_data, _ = cls.load_dataset()
        num_shards, num_texts = 200, len(train_data) // 200
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(num_users)}
        idxs = np.arange(num_shards * num_texts)

        labels = np.random.randint(0, num_users, size=len(train_data))

        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_texts:(rand + 1) * num_texts]), axis=0)
        return dict_users