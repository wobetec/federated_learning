from argparse import Namespace
from src.experiments import BaseLine, Federated
import torch
from src.save import save_results
import pandas as pd


def run_experiment(experiment):
    experiment = Namespace(**experiment)
    if experiment.exp_name == 'baseline':
        exp = BaseLine(experiment)
        exp.train()
    elif experiment.exp_name == 'federated':
        exp = Federated(experiment)
        exp.train()
    else:
        raise ValueError(f'Unrecognized experiment name: {experiment.exp_name}')
    save_results(exp)

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        gpu = 0
    else:
        print("No GPU available. Training will run on CPU.")
        gpu = None

    df = pd.read_csv('paper_models.csv')
    scheduled_experiments = df.to_dict(orient='records')

    for experiment in (scheduled_experiments):
        experiment['gpu'] = gpu
        experiment['verbose'] = 1
        run_experiment(experiment)

