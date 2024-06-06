import json
import os

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
    if hasattr(experiment, '__dict__'):
        exp_dict = experiment.__dict__()  # Chama o método __dict__ personalizado
    else:
        exp_dict = experiment
    
    # Cria uma chave única baseada nos parâmetros do experimento
    exp_key = f"{exp_dict['exp_name']}_{exp_dict['model_name']}_{exp_dict['dataset_name']}_{exp_dict['lr']}_{exp_dict['epochs']}"

    if exp_key not in results:
        results[exp_key] = exp_dict

        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Experimento salvo com sucesso: {exp_key}")
    else:
        print(f"Experimento já existe: {exp_key}")
