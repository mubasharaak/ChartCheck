import json
import os

import evaluate
import numpy as np
import pandas as pd

import torch
from transformers import Trainer, TapasForSequenceClassification, TapasTokenizer, TapasConfig
from transformers import TrainingArguments
from sklearn.metrics import f1_score

# variables
MAX_LENGTH = 1024
DATASET_PATH_TRAIN = "dataset/claim_explanation_verification_pre_tasksets_train_V2.json"
DATASET_PATH_VAL = "dataset/claim_explanation_verification_pre_tasksets_validation_V2.json"
DATASET_PATH_TEST = "dataset/claim_explanation_verification_pre_tasksets_test_V2.json"
DATASET_PATH_TEST_TWO = "dataset/claim_explanation_verification_pre_tasksets_test_two_V2.json"

LABEL_DICT = {
    "Yes": 0,
    "No": 1,
    "TRUE": 0,
    "FALSE": 1,
}
METRIC = evaluate.load("glue", "mrpc")


def join_unicode(delim, entries):
    return delim.join(entries)


def _read_chart_dataset(dataset):
    claims = []
    tables = []
    labels = []

    for item in dataset:
        path_table = os.path.join("/scratch/users/k20116188/chart-fact-checking/deplot-tables",
                                  os.path.basename(item["chart_img"]) + ".txt")
        try:
            with open(path_table, "r", encoding="utf-8") as f:
                table = f.read().splitlines()

            claim = str(item["claim"])
            caption = str(item["caption"])
            label = str(item["label"])
        except IndexError as e:
            print(f"Exception for file {path_table}: {e}.")
            continue

        # claims.append(claim + "[SEP]" + caption)
        claims.append(claim)
        tables.append(table)
        labels.append(LABEL_DICT[label])

    return claims, tables, labels


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def _compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    f1_micro = f1_score(y_true=labels, y_pred=predictions, average='micro')
    f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')
    return {"f1_micro": f1_micro, "f1_macro": f1_macro}


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, claims, tables, labels, tokenizer):
        self.claims = claims
        self.tables = tables
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        table = self.tables[idx]
        claim = self.claims[idx]
        label = self.labels[idx]

        # convert to tabel (as a pd.DataFrame)
        for i, entry in enumerate(table):
            table[i] = str(table[i]).split(" | ")
        try:
            table_list = [len(entry) for entry in table if entry and entry != None]
            if table_list:
                max_len = max(table_list)
                for i, entry in enumerate(table):
                    if entry and entry!=None:
                        while len(entry) < max_len:
                            entry.append("")
                        table[i] = entry
                    else:
                        table[i] = ["" for i in range(max_len)]
                if len(table)>2:
                    table_df = pd.DataFrame(table[2:], columns=table[1]).astype(str)
                else:
                    table_df = pd.DataFrame({'': []}).astype(str)
            else:
                table_df = pd.DataFrame({'': []}).astype(str)
        except Exception as e:
            print(f"Following exception occurred for table {table}: {e}")

        encoding = self.tokenizer(table=table_df,
                                  queries=[claim],
                                  padding="max_length",
                                  truncation=True,
                                  return_tensors="pt",

                                  )
        # remove the batch dimension which the tokenizer adds by default
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels'] = torch.tensor([label])

        return encoding

    def __len__(self):
        return len(self.claims)


def train(model, train_dataset, dev_dataset, test_dataset, save_path, only_test=False):
    # Train model
    trainer = model_trainer(
        model,
        train_dataset,
        dev_dataset,
        training_epochs=8,
        batch_size=16,
        learning_rate=5e-5)

    if not only_test:
        trainer.train()
        trainer.save_model(save_path)

    result_dict = trainer.predict(test_dataset)
    print(result_dict.metrics)
    return result_dict


def continue_training(model, training_args, train_dataset, dev_dataset, test_dataset):
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=dev_dataset,             # evaluation dataset
        compute_metrics=_compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model("./results/chart_table_classification")

    result_dict = trainer.predict(test_dataset)
    print(result_dict.metrics)
    return result_dict


def model_trainer(model, train_dataset, dev_dataset, training_epochs, batch_size, learning_rate):
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=batch_size,
        # weight_decay=weight_decay,
        num_train_epochs=training_epochs,
        # learning_rate=learning_rate,
        per_device_eval_batch_size=64,
        evaluation_strategy="steps",
        eval_steps=100,
        gradient_accumulation_steps=4,
        disable_tqdm=False,
        fp16=True,
        save_total_limit=1,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=_compute_metrics
    )

    return trainer


def _load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _create_dataset(path: str, tok):
    input_data = _load_dataset(path)
    claims, tables, labels = _read_chart_dataset(input_data)
    return TableDataset(claims, tables, labels, tok)


if __name__ == "__main__":
    # Load model
    hg_model_hub_name = "google/tapas-base-finetuned-tabfact"
    tokenizer = TapasTokenizer.from_pretrained(hg_model_hub_name)
    model = TapasForSequenceClassification.from_pretrained("./results/chart_table_classification_TAPAS", torch_dtype="auto")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    train_dataset = _create_dataset(DATASET_PATH_TRAIN, tokenizer)
    test_dataset = _create_dataset(DATASET_PATH_TEST, tokenizer)
    test_two_dataset = _create_dataset(DATASET_PATH_TEST_TWO, tokenizer)
    eval_dataset = _create_dataset(DATASET_PATH_VAL, tokenizer)

    output_dir = "./results/chart_table_classification_TAPAS"
    results_dict = train(model, train_dataset=train_dataset,
                         dev_dataset=eval_dataset, test_dataset=test_dataset, only_test=True, save_path=output_dir)

    results_dict_two = train(model, train_dataset=train_dataset,
                         dev_dataset=eval_dataset, test_dataset=test_two_dataset, only_test=True, save_path=output_dir)

    print(f"Saving output to: {output_dir}.")

    with open(os.path.join(output_dir, "test_output.txt"), "w") as f:
        f.write(f"metrics: {results_dict.metrics}")
        for i, logits in enumerate(results_dict.predictions.tolist()):
            predictions = np.argmax(logits, axis=-1)
            f.write(f"input: {tokenizer.decode(test_dataset[i]['input_ids'])}\n")
            f.write(f"label: {LABEL_DICT[results_dict.label_ids[i]]}\n")
            f.write(f"prediction: {LABEL_DICT[predictions[i]]}\n\n")

    with open(os.path.join(output_dir, "output_test_two.txt"), "w") as f:
        f.write(f"metrics: {results_dict_two.metrics}")
        for i, logits in enumerate(results_dict_two.predictions.tolist()):
            predictions = np.argmax(logits, axis=-1)
            f.write(f"input: {tokenizer.decode(test_dataset[i]['input_ids'])}\n")
            f.write(f"label: {LABEL_DICT[results_dict_two.label_ids[i]]}\n")
            f.write(f"prediction: {LABEL_DICT[predictions[i]]}\n\n")



