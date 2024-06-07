from argparse import Namespace
from src.experiments import BaseLine, Federated
import torch
from src.save import save_results, experiment_exists

scheduled_experiments = [
    {
        'exp_name': 'baseline',

        'epochs': 2,
        'lr': 0.01,

        'model': 'cnn',
        'optimizer': 'sgd',

        'dataset': 'mnist',
        'num_classes': 10,
        'iid': 1,
        'unequal': 0,
    },
    # {
    #     'exp': 'federated',

    #     'epochs': '',
    #     'num_users': '',
    #     'frac': '',
    #     'local_ep': '',
    #     'local_bs': '',
    #     'lr': '',

    #     'model': '',
    #     'optimizer': '',

    #     'dataset': '',
    #     'num_classes': '',
    #     'iid': '',
    #     'unequal': '',
    # },
]

def run_experiment(experiment):
    experiment = Namespace(**experiment)
    if experiment.exp_name == 'baseline':
        exp = BaseLine(experiment)
    elif experiment.exp_name == 'federated':
        exp = Federated(experiment)
    else:
        raise ValueError(f'Unrecognized experiment name: {experiment.exp_name}')

    if experiment_exists(exp):
        print(f"Experimento já existe: {experiment.exp_name}")
        return

    exp.train()
    save_results(exp)

if __name__ == '__main__':

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        gpu = None
    else:
        print("No GPU available. Training will run on CPU.")
        gpu = None
    for experiment in scheduled_experiments:
        experiment['gpu'] = gpu
        experiment['verbose'] = 1
        run_experiment(experiment)

