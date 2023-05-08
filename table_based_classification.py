import argparse
import csv
import json
import os
import random
import re
import sys
from collections import OrderedDict

import evaluate
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import TrainingArguments
from sklearn.metrics import f1_score

# variables
max_length = 1024
train_dataset_path = os.path.join("data", "")
test_dataset_path = os.path.join("data", "")
dev_dataset_path = os.path.join("data", "")

DATASET_PATH = "claim_explanation_verification_pre_tasksets.json"

label_dict = {
    "Yes": 0,
    "No": 1,
    "TRUE": 0,
    "FALSE": 1,
}


# functions
def join_unicode(delim, entries):
    return delim.join(entries)


def read_chart_dataset(dataset):
    claims = []
    tables = []
    labels = []

    for item in dataset:
        try:
            path_table = os.path.join("/scratch/users/k20116188/chart-fact-checking/deplot-tables",
                                      os.path.basename(item["chart_img"]) + ".txt")
            with open(path_table, "w", encoding="utf-8") as f:
                table = f.readlines()

            claim = item["claim"]
            label = label_dict[item["label"]]
        except IndexError:
            continue

        claims.append(claim)
        tables.append(table)
        labels.append(label_dict[label])

    return claims, tables, labels


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    f1_micro = f1_score(y_true=labels, y_pred=predictions, average='micro')
    f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')
    return {"f1_micro": f1_micro, "f1_macro": f1_macro}


class ChartDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


bs = 2
num_epochs = 15
metric = evaluate.load("glue", "mrpc")

training_args = TrainingArguments(
    output_dir='./results/chart_table_classification',          # output directory
    num_train_epochs=num_epochs,              # total number of training epochs
    per_device_train_batch_size=bs,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=50,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    metric_for_best_model="eval_f1_micro",
    save_total_limit=1,
    load_best_model_at_end=True,
    learning_rate=1e-06,
    fp16=True,                        # mixed precision training
)


def train(model, training_args, train_dataset, dev_dataset, test_dataset, only_test=False):
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
    )

    if not only_test:
        trainer.train()
        trainer.save_model("./results/chart_table_classification")

    result_dict = trainer.predict(test_dataset)
    print(result_dict.metrics)
    return result_dict


def continue_training(model, training_args, train_dataset, dev_dataset, test_dataset):
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model("./results/chart_table_classification")

    result_dict = trainer.predict(test_dataset)
    print(result_dict.metrics)
    return result_dict


if __name__ == "__main__":
    # Load model
    hg_model_hub_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name, torch_dtype="auto")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    # load file
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    np.random.seed(42)

    # Shuffle the indices of the data
    indices = np.random.permutation(len(data))

    # Calculate the number of samples in the training, validation, and testing sets
    num_train = int(0.8 * len(data))
    num_val = int(0.1 * len(data))

    # Split the indices into training, validation, and testing sets
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    train_data = data[train_indices]
    val_data = data[val_indices]
    test_data = data[test_indices]

    # Dataset preperation
    train_claims, train_tables, train_labels = read_chart_dataset(train_data)
    test_claims, test_tables, test_labels = read_chart_dataset(test_data)
    val_claims, val_tables, val_labels = read_chart_dataset(val_data)

    train_tokenized = tokenizer(train_claims, train_tables,
                                max_length=max_length,
                                return_token_type_ids=True, truncation=True,
                                padding=True)
    train_dataset = ChartDataset(train_tokenized, train_labels)
    test_tokenized = tokenizer(test_claims, test_tables,
                               max_length=max_length,
                               return_token_type_ids=True, truncation=True,
                               padding=True)
    test_dataset = ChartDataset(test_tokenized, test_labels)
    dev_tokenized = tokenizer(val_claims, val_tables,
                              max_length=max_length,
                              return_token_type_ids=True, truncation=True,
                              padding=True)
    dev_dataset = ChartDataset(dev_tokenized, val_labels)

    results_dict = train(model, training_args, train_dataset=train_dataset,
                         dev_dataset=dev_dataset, test_dataset=test_dataset, only_test=False)

    with open("./results/deberta_classification_output.txt", "w") as f:
        for i, logits in enumerate(results_dict.predictions.tolist()):
            predictions = np.argmax(logits, axis=-1)
            if predictions != results_dict.label_ids.tolist()[i]:
                f.write(f"input: {tokenizer.decode(test_dataset[i]['input_ids'])}\n")
                f.write(f"label: {label_dict[results_dict.label_ids.tolist()[i]]}\n")
                f.write(f"prediction: {label_dict[predictions]}\n\n")
