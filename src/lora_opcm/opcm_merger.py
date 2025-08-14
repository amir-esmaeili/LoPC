# src/lora_opcm/opcm_merger.py

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
import os
from peft import PeftModel


@dataclass
class LoRAModule:
    """Container for LoRA parameters of a single layer."""
    lora_B: torch.Tensor
    lora_A: torch.Tensor
    alpha: float
    rank: int


@dataclass
class MergedLoRAState:
    """State of merged LoRA modules for a single layer."""
    B_merged: torch.Tensor
    A_merged: torch.Tensor
    # Other state variables from pseudo-code can be added here
    task_count: int = 0


class LoRAOPCMMerger:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    def _get_lora_state_dict(self, model_path):
        """Loads the LoRA adapter's state dictionary."""
        # This assumes the base model is not loaded, which is memory efficient
        return torch.load(os.path.join(model_path, 'adapter_model.bin'), map_location=self.device)

    def _extract_lora_modules(self, state_dict, target_modules) -> Dict[str, LoRAModule]:
        """Extracts LoRA B and A matrices for target layers from a state dict."""
        modules = {}
        for layer_name in target_modules:
            # PEFT stores weights like: base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.default.weight
            # We need to find the full key name.
            lora_A_key = next((k for k in state_dict if layer_name in k and 'lora_A' in k), None)
            lora_B_key = next((k for k in state_dict if layer_name in k and 'lora_B' in k), None)

            if lora_A_key and lora_B_key:
                modules[layer_name] = LoRAModule(
                    lora_B=state_dict[lora_B_key],
                    lora_A=state_dict[lora_A_key],
                    alpha=self.config.model.lora_alpha,
                    rank=self.config.model.lora_rank
                )
        return modules

    def _orthogonal_projection(self, new_module: LoRAModule, merged_state: MergedLoRAState) -> LoRAModule:
        """
        Projects the new LoRA module to be orthogonal to the merged state.
        This is a simplified implementation based on your pseudo-code.
        """
        B_new, A_new = new_module.lora_B, new_module.lora_A
        B_merged, A_merged = merged_state.B_merged, merged_state.A_merged

        # SVD of the merged matrices (as per your pseudo-code)
        U_B, S_B, Vt_B = torch.linalg.svd(B_merged, full_matrices=False)
        U_A, S_A, Vt_A = torch.linalg.svd(A_merged.T, full_matrices=False)  # SVD on A.T for column space

        # Project B_new to be orthogonal to the column space of B_merged
        proj_B_on_merged = U_B @ U_B.T @ B_new
        B_proj = B_new - proj_B_on_merged

        # Project A_new to be orthogonal to the column space of A_merged.T (row space of A_merged)
        proj_A_on_merged = U_A @ U_A.T @ A_new
        A_proj = A_new - proj_A_on_merged

        return LoRAModule(lora_B=B_proj, lora_A=A_proj, alpha=new_module.alpha, rank=new_module.rank)

    def merge_adapters(self, adapter_paths: List[str]) -> Dict[str, MergedLoRAState]:
        """
        Main merging algorithm to sequentially merge LoRA adapters.
        """
        print("\n--- Starting LoRA-OPCM Merging Process ---")

        # This will hold the final merged state for each layer
        final_merged_layers: Dict[str, MergedLoRAState] = {}

        # 1. Initialize with the first adapter
        print(f"Initializing with adapter: {adapter_paths[0]}")
        first_adapter_state_dict = self._get_lora_state_dict(adapter_paths[0])
        initial_modules = self._extract_lora_modules(first_adapter_state_dict, self.config.model.target_modules)

        for layer_name, module in initial_modules.items():
            final_merged_layers[layer_name] = MergedLoRAState(
                B_merged=module.lora_B.clone(),
                A_merged=module.lora_A.clone(),
                task_count=1
            )

        # 2. Sequentially merge the remaining adapters
        for i, adapter_path in enumerate(adapter_paths[1:], start=1):
            print(f"Merging adapter {i + 1}/{len(adapter_paths)}: {adapter_path}")
            new_adapter_state_dict = self._get_lora_state_dict(adapter_path)
            new_modules = self._extract_lora_modules(new_adapter_state_dict, self.config.model.target_modules)

            for layer_name, new_module in new_modules.items():
                merged_state = final_merged_layers[layer_name]

                # Step 2: Compute orthogonal projection (from your pseudo-code)
                projected_module = self._orthogonal_projection(new_module, merged_state)

                # Step 3 & 4: Simple averaging for this proof-of-concept
                # Your pseudo-code's adaptive scaling can be implemented here later
                current_task_count = merged_state.task_count

                # Update merged parameters
                merged_state.B_merged = (current_task_count * merged_state.B_merged + projected_module.lora_B) / (
                            current_task_count + 1)
                merged_state.A_merged = (current_task_count * merged_state.A_merged + projected_module.lora_A) / (
                            current_task_count + 1)
                merged_state.task_count += 1

        print("--- Merging Complete ---")
        return final_merged_layers