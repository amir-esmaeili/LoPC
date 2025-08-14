# src/lora_opcm/evaluation.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class MultiTaskEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.base_name)

    def _create_peft_model_with_merged_weights(self, num_labels, merged_layers):
        """Creates a PEFT model and injects the merged LoRA weights into it."""
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.base_name,
            num_labels=num_labels
        ).to(self.device)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.config.model.lora_rank,
            lora_alpha=self.config.model.lora_alpha,
            target_modules=list(self.config.model.target_modules)
        )
        peft_model = get_peft_model(base_model, peft_config)

        # This is the crucial part: update the model's state with our merged weights
        with torch.no_grad():
            for layer_name, merged_state in merged_layers.items():
                # Find the corresponding layers in the new peft_model
                key_A = f"base_model.model.bert.encoder.layer.0.attention.self.{layer_name}.lora_A.default.weight"
                key_B = f"base_model.model.bert.encoder.layer.0.attention.self.{layer_name}.lora_B.default.weight"

                # Find the actual keys (they might differ slightly based on model architecture)
                full_key_A = next((k for k in peft_model.state_dict() if layer_name in k and 'lora_A' in k), None)
                full_key_B = next((k for k in peft_model.state_dict() if layer_name in k and 'lora_B' in k), None)

                if full_key_A and full_key_B:
                    peft_model.state_dict()[full_key_A].copy_(merged_state.A_merged)
                    peft_model.state_dict()[full_key_B].copy_(merged_state.B_merged)

        return peft_model

    def evaluate(self, merged_layers):
        """Evaluates the merged model on all tasks defined in the config."""
        print("\n--- Starting Evaluation of Merged Model ---")
        results = {}

        for task_config in self.config.tasks:
            print(f"Evaluating on task: {task_config.task_name}")

            # 1. Load dataset and determine num_labels
            dataset = load_dataset(task_config.dataset_name, task_config.dataset_config)
            num_labels = dataset['train'].features['label'].num_classes
            task_config.num_labels = num_labels  # Store for trainer

            # 2. Create a fresh model with the merged weights
            model = self._create_peft_model_with_merged_weights(num_labels, merged_layers)
            model.eval()

            # 3. Prepare dataloader for the test set
            def preprocess_function(examples):
                if isinstance(task_config.text_column, list):
                    return self.tokenizer(*[examples[col] for col in task_config.text_column], truncation=True,
                                          padding="max_length", max_length=128)
                else:
                    return self.tokenizer(examples[task_config.text_column], truncation=True, padding="max_length",
                                          max_length=128)

            # Use validation set if test set is not available/has no labels (like GLUE)
            eval_split = 'validation' if task_config.dataset_config != 'mnli' else 'validation_matched'
            if 'test' in dataset and task_config.task_name not in ['sst2', 'rte', 'mrpc']:
                eval_split = 'test'

            tokenized_eval_dataset = dataset[eval_split].map(preprocess_function, batched=True)
            tokenized_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

            dataloader = torch.utils.data.DataLoader(tokenized_eval_dataset, batch_size=32)

            # 4. Run inference
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in dataloader:
                    labels = batch['label'].to(self.device)
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}

                    outputs = model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=-1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # 5. Compute metrics
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            results[task_config.task_name] = {"accuracy": accuracy, "f1": f1}
            print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

        print("--- Evaluation Complete ---")
        return results