import json
import os

def get_name(exp_dict):
    if exp_dict['exp_name'] == 'baseline':
        return f"{exp_dict['exp_name']}_{exp_dict['model_name']}_{exp_dict['dataset_name']}_{exp_dict['lr']}_{exp_dict['epochs']}_{exp_dict['optimizer_name']}_{exp_dict['num_classes']}_{exp_dict['iid']}_{exp_dict['unequal']}"
    else:
        return f"{exp_dict['exp_name']}_{exp_dict['model_name']}_{exp_dict['dataset_name']}_{exp_dict['lr']}_{exp_dict['epochs']}_{exp_dict['num_users']}_{exp_dict['frac']}_{exp_dict['local_ep']}_{exp_dict['local_bs']}_{exp_dict['optimizer_name']}_{exp_dict['num_classes']}_{exp_dict['iid']}_{exp_dict['unequal']}"

def experiment_exists(experiment, filename="results.json", save_dir="save"):
    file_path = os.path.join(save_dir, filename)

    if not os.path.exists(file_path):
        return False

    with open(file_path, "r") as f:
        results = json.load(f)

    exp_dict = experiment.to_dict()
    exp_key = get_name(exp_dict)

    return exp_key in results

def save_results(experiment, filename="results.json", save_dir="save"):
    # Cria o diretório se não existir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, filename)

    results = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            results = json.load(f)

    # Usa o dicionário interno do objeto se for um objeto
    exp_dict = experiment.to_dict()
    
    # Cria uma chave única baseada nos parâmetros do experimento
    exp_key = get_name(exp_dict)

    if exp_key not in results:
        results[exp_key] = exp_dict

        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Experimento salvo com sucesso: {exp_key}")
    else:
        print(f"Experimento não salvo, já existe: {exp_key}")
