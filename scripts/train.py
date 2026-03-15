import os
import pandas as pd
import torch
import yaml
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
import evaluate
import numpy as np
import sacrebleu
import math

from datetime import datetime

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Fill NaN with empty strings
    train_df['transliteration'] = train_df['transliteration'].fillna('')
    train_df['translation'] = train_df['translation'].fillna('')
    val_df['transliteration'] = val_df['transliteration'].fillna('')
    val_df['translation'] = val_df['translation'].fillna('')

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    return train_dataset, val_dataset

def preprocess_function(examples, tokenizer, max_length):
    inputs = examples["transliteration"]
    targets = examples["translation"]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=max_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class ValidationCallback(TrainerCallback):
    def __init__(self, tokenizer, config, experiment_dir):
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = experiment_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        pass

# Evaluation Metrics are now handled directly in compute_metrics via sacrebleu

def compute_metrics(eval_pred, tokenizer, config, experiment_dir, state=None, dataset=None):
    preds, labels = eval_pred
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()
    
    # Filter out negative values to prevent OverflowError in tokenizers
    if isinstance(preds, list):
        preds = [[t for t in seq if t >= 0] for seq in preds]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    hypotheses = [pred.strip() for pred in decoded_preds]
    references = [label.strip() for label in decoded_labels]

    # Calculate BLEU using sacrebleu directly (matches snippet)
    bleu_res = sacrebleu.corpus_bleu(hypotheses, [references])
    
    # Calculate CHRF++ (word_order=2) (matches snippet)
    chrf_res = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    
    bleu_score = bleu_res.score
    chrf_score = chrf_res.score
    geo_mean = math.sqrt(bleu_score * chrf_score) if (bleu_score * chrf_score) > 0 else 0

    res = {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "geo_mean": geo_mean
    }

    # Save validation samples and track metrics in CSV
    if state is not None:
        step = state.global_step
        
        # 1. Save Samples CSV
        samples_file = os.path.join(experiment_dir, f"val_samples_step_{step}.csv")
        sample_data = {
            'prediction': decoded_preds,
            'reference': decoded_labels
        }
        if dataset is not None:
            # Match the number of samples
            sample_data['input'] = dataset['transliteration'][:len(decoded_preds)]
        
        pd.DataFrame(sample_data).to_csv(samples_file, index=False)
        print(f"\n--- Saved validation samples to {samples_file} ---")

        # 2. Update Metrics History CSV
        history_file = os.path.join(experiment_dir, "metrics_history.csv")
        history_row = {'step': step, 'epoch': state.epoch}
        history_row.update(res)
        
        history_df = pd.DataFrame([history_row])
        if os.path.exists(history_file):
            history_df.to_csv(history_file, mode='a', header=False, index=False)
        else:
            history_df.to_csv(history_file, index=False)

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--experiment_name", type=str, default="experiment")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Generate Dynamic Experiment Path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = config['model']['name'].split('/')[-1]
    lr = config['training']['learning_rate']
    bs = config['training']['per_device_train_batch_size']
    acc = config['training']['gradient_accumulation_steps']
    
    # Sanitize experiment name for Windows
    sanitized_exp_name = args.experiment_name.strip().replace('\n', '').replace('\r', '')
    # Basic character replacement for common invalid filename chars
    for char in ['*', '?', '"', '<', '>', '|', ':']:
        sanitized_exp_name = sanitized_exp_name.replace(char, '_')
        
    experiment_dir = os.path.join("experiments", f"{timestamp}_{sanitized_exp_name}_{model_name_safe}_lr{lr}_bs{bs}x{acc}")
    logging_dir = os.path.join(experiment_dir, "logs")
    os.makedirs(experiment_dir, exist_ok=True)

    print("--- Akkadian Training Script Initialized ---")
    print(f"Experiment Directory: {experiment_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    train_dataset, val_dataset = load_data(config['data']['train_path'], config['data']['val_path'])
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'], 
        src_lang=config['model']['source_lang'], 
        tgt_lang=config['model']['target_lang']
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model']['name'])

    # Prepare data
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config['model']['max_length']), 
        batched=True
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config['model']['max_length']), 
        batched=True
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=experiment_dir,
        eval_strategy=config['training']['eval_strategy'],
        save_strategy=config['training']['save_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        learning_rate=float(config['training']['learning_rate']),
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        weight_decay=config['training']['weight_decay'],
        save_total_limit=config['training']['save_total_limit'],
        num_train_epochs=config['training']['num_train_epochs'],
        predict_with_generate=True,
        fp16=config['training']['fp16'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        logging_dir=logging_dir,
        logging_steps=config['training']['logging_steps'],
        push_to_hub=False,
        report_to=["tensorboard"],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        gradient_checkpointing=False
    )

    class CustomTrainer(Seq2SeqTrainer):
        pass

    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(
            eval_pred, 
            tokenizer, 
            config, 
            experiment_dir,
            trainer.state, 
            trainer.eval_dataset
        )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper
    )

    # Add Callbacks
    trainer.add_callback(ValidationCallback(tokenizer, config, experiment_dir))

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(experiment_dir, "final"))
    print(f"Training complete. Model saved to {experiment_dir}/final")
