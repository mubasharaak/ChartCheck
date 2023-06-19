import json
import os

import evaluate
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import TrainingArguments
from sklearn.metrics import f1_score

# variables
max_length = 1024

DATASET_PATH_TRAIN = "dataset/claim_explanation_verification_pre_tasksets_train_V2.json"
DATASET_PATH_VAL = "dataset/claim_explanation_verification_pre_tasksets_validation_V2.json"
# DATASET_PATH_TEST = "dataset/claim_explanation_verification_pre_tasksets_test_V2.json"
# DATASET_PATH_TEST_TWO = "dataset/claim_explanation_verification_pre_tasksets_test_two_V2.json"

DATASET_PATH_TEST = "dataset/claim_explanation_test_one_ID_reasoning.json" # manually annotated subset
DATASET_PATH_TEST_TWO = "dataset/claim_explanation_test_two_ID_reasoning.json" # manually annotated subset

label_dict = {
    "Yes": 0,
    "No": 1,
    "TRUE": 0,
    "FALSE": 1,
}

label_dict_reverse = {
    "0": "True",
    "1": "False",
    "2": "NEI",
}


# functions
def join_unicode(delim, entries):
    return delim.join(entries)


def read_chart_dataset(dataset):
    claims = []
    evidences = []
    labels = []

    for item in dataset:
        try:
            path_table = os.path.join("/scratch/users/k20116188/chart-fact-checking/deplot-tables",
                                      os.path.basename(item["chart_img"]) + ".txt")

            with open(path_table, "r", encoding="utf-8") as f:
                table = str(f.readlines())

            claim = str(item["claim"])
            label = str(item["label"])
            caption = str(item["caption"])

            # table = "" # using for the hypothesis-only baseline
            # caption = "" # using for the hypothesis-only baseline
        except IndexError as e:
            print(f"Exception for file {path_table}: {e}.")
            continue

        claims.append(claim)
        evidences.append(table + ". " + caption)
        labels.append(label_dict[label])

    return claims, evidences, labels


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
num_epochs = 3
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
    eval_steps=300,
    save_steps=300,
    metric_for_best_model="eval_f1_micro",
    save_total_limit=1,
    load_best_model_at_end=True,
    learning_rate=1e-06,
    fp16=True,                        # mixed precision training
)


def train(model, training_args, train_dataset, dev_dataset, test_dataset, save_path, only_test=False):
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
    )

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
    # hg_model_hub_name = "./results/chart_table_classification_DeBERTa"
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained("./results/chart_table_classification_DeBERTa", torch_dtype="auto")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    # load files (train, validation, first and second test set)
    with open(DATASET_PATH_TRAIN, "r", encoding="utf-8") as file:
        train_data = json.load(file)

    with open(DATASET_PATH_VAL, "r", encoding="utf-8") as file:
        val_data = json.load(file)

    with open(DATASET_PATH_TEST, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    with open(DATASET_PATH_TEST_TWO, "r", encoding="utf-8") as file:
        test_two_data = json.load(file)

    # Dataset preperation
    train_claims, train_tables, train_labels = read_chart_dataset(train_data)
    test_claims, test_tables, test_labels = read_chart_dataset(test_data)
    test_two_claims, test_two_tables, test_two_labels = read_chart_dataset(test_two_data)
    val_claims, val_tables, val_labels = read_chart_dataset(val_data)

    print(f"Number of test samples {len(test_claims)}")
    print(f"Number of test two samples {len(test_two_claims)}")

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
    test_two_tokenized = tokenizer(test_two_claims, test_two_tables,
                               max_length=max_length,
                               return_token_type_ids=True, truncation=True,
                               padding=True)
    test_two_dataset = ChartDataset(test_two_tokenized, test_two_labels)
    dev_tokenized = tokenizer(val_claims, val_tables,
                              max_length=max_length,
                              return_token_type_ids=True, truncation=True,
                              padding=True)
    dev_dataset = ChartDataset(dev_tokenized, val_labels)

    output_dir = "./results/chart_table_classification_DeBERTa_REASONING_SUBSET"
    # Check if the directory exists, otherwise create
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_dict = train(model, training_args, train_dataset=train_dataset,
                         dev_dataset=dev_dataset, test_dataset=test_dataset, only_test=True, save_path=output_dir)

    print(f"len(results_dict.predictions.tolist()): {len(results_dict.predictions.tolist())}")

    results_dict_two = train(model, training_args, train_dataset=train_dataset,
                             dev_dataset=dev_dataset, test_dataset=test_two_dataset, only_test=True, save_path=output_dir)
    print(f"len(results_dict_two.predictions.tolist()): {len(results_dict_two.predictions.tolist())}")

    with open(os.path.join(output_dir, "output_test.txt"), "w") as f:
        f.write(f"result_dict.metrics: {results_dict.metrics}\n\n")

        for i, logits in enumerate(results_dict.predictions.tolist()):
            predictions = np.argmax(logits, axis=-1)
            if 'claim_id' in test_data[i].keys():
                f.write(f"claim_id: {test_data[i]['claim_id']}\n")
            f.write(f"input: {tokenizer.decode(test_dataset[i]['input_ids'])}\n")
            f.write(f"label: {label_dict_reverse[str(results_dict.label_ids.tolist()[i])]}\n")
            f.write(f"prediction: {label_dict_reverse[str(predictions)]}\n\n")

    with open(os.path.join(output_dir, "output_test_two.txt"), "w") as f:
        f.write(f"result_dict.metrics: {results_dict_two.metrics}\n\n")

        for i, logits in enumerate(results_dict_two.predictions.tolist()):
            predictions = np.argmax(logits, axis=-1)
            if 'claim_id' in test_two_data[i].keys():
                f.write(f"claim_id: {test_two_data[i]['claim_id']}\n")
            f.write(f"input: {tokenizer.decode(test_two_dataset[i]['input_ids'])}\n")
            f.write(f"label: {label_dict_reverse[str(results_dict_two.label_ids.tolist()[i])]}\n")
            f.write(f"prediction: {label_dict_reverse[str(predictions)]}\n\n")
