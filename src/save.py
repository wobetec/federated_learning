import json
import os

def save_results(experiment, filename="results.json", save_dir="saved_results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, filename)

    results = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            results = json.load(f)

    exp_key = f"{experiment['exp_name']}_{experiment['model']}_{experiment['dataset']}_{experiment['lr']}_{experiment['epochs']}"

    if exp_key not in results:
        results[exp_key] = experiment

        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Experimento salvo: {exp_key}")
    else:
        print(f"Experimento já existe: {exp_key}")
