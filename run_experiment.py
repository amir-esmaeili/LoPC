# run_experiment.py (Corrected Version 2)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import random
import os
import json

from src.lora_opcm.lora_trainer import train_lora_for_task
from src.lora_opcm.opcm_merger import LoRAOPCMMerger
from src.lora_opcm.evaluation import MultiTaskEvaluator
from datasets import load_dataset


@hydra.main(version_base=None, config_path="experiments/configs", config_name="proof_of_concept")
def main(cfg: DictConfig):
    # --- 1. Setup ---
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    os.makedirs(cfg.training.output_dir, exist_ok=True)

    trained_adapter_paths = []

    # --- 2. Train individual LoRA adapters for each task ---
    for task_config in cfg.tasks:
        print(f"\n{'=' * 20} PREPARING TASK: {task_config.task_name.upper()} {'=' * 20}")

        # FIX 1: Removed `trust_remote_code=True`
        print(f"Loading dataset '{task_config.dataset_name}' with config '{task_config.dataset_config}'...")
        dataset = load_dataset(task_config.dataset_name, task_config.dataset_config)

        num_labels = dataset['train'].features[task_config.label_column].num_classes
        print(f"Task '{task_config.task_name}' has {num_labels} labels.")

        # FIX 2: Pass num_labels as a separate argument instead of modifying the config
        adapter_path = train_lora_for_task(cfg, task_config, num_labels)
        trained_adapter_paths.append(adapter_path)

    # --- 3. Merge the trained adapters using LoRA-OPCM ---
    merger = LoRAOPCMMerger(cfg)
    merged_layers = merger.merge_adapters(trained_adapter_paths)

    # --- 4. Evaluate the merged model ---
    evaluator = MultiTaskEvaluator(cfg)
    results = evaluator.evaluate(merged_layers)

    # --- 5. Display and Save Final Results ---
    print("\n\n" + "=" * 25 + " FINAL RESULTS " + "=" * 25)

    output_dir = os.getcwd()
    print(f"Output directory for this run: {output_dir}")
    print(json.dumps(results, indent=2))

    results_path = os.path.join(output_dir, "final_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()