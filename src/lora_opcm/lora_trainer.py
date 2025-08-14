# src/lora_opcm/lora_trainer.py

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import os


def train_lora_for_task(config, task_config, num_labels):
    """
    Trains a LoRA model for a single task specified in the config.
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.base_name,
        num_labels=num_labels
    ).to(device)

    # 2. Load and Preprocess Dataset
    dataset = load_dataset(task_config.dataset_name, task_config.dataset_config)

    def preprocess_function(examples):
        # This handles both single-sentence and sentence-pair tasks dynamically
        text_cols = task_config.text_column
        inputs = [examples[col] for col in text_cols] if isinstance(text_cols, list) else [examples[text_cols]]
        return tokenizer(*inputs, truncation=True, padding="max_length", max_length=128)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column(task_config.label_column, "labels")

    # 3. Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=0.1,
        target_modules=list(config.model.target_modules)
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Set up Trainer with modern arguments
    output_path = os.path.join(config.training.output_dir, task_config.task_name)

    # The arguments `evaluation_strategy` and `save_strategy` should be compatible
    # with the latest `transformers` library after you upgrade.
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        eval_strategy="epoch",  # This will now be recognized
        save_strategy="epoch",  # Aligning save strategy with evaluation
        logging_dir=f"{output_path}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # Use 'loss', 'accuracy', or 'f1'
        greater_is_better=False  # Set to True if metric_for_best_model is accuracy/f1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # 5. Train
    print(f"\n--- Starting LoRA training for task: {task_config.task_name} ---")
    trainer.train()

    # 6. Save the trained LoRA adapter
    adapter_path = os.path.join(output_path, "best_adapter")
    model.save_pretrained(adapter_path)
    print(f"--- Finished training. Best LoRA adapter saved to {adapter_path} ---")

    return adapter_path