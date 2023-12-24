import json
import os

import evaluate
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import TrainingArguments

# variables
MAX_LENGTH = 1024

DATASET_PATH_TRAIN = "dataset/claim_explanation_verification_pre_tasksets_train_V2.json"
DATASET_PATH_VAL = "dataset/claim_explanation_verification_pre_tasksets_validation_V2.json"
# DATASET_PATH_TEST = "dataset/claim_explanation_verification_pre_tasksets_test_V2.json"
# DATASET_PATH_TEST_TWO = "dataset/claim_explanation_verification_pre_tasksets_test_two_V2.json"

# manually annotated reasoning subset
# DATASET_PATH_TEST = "dataset/claim_explanation_test_one_ID_reasoning.json"
# DATASET_PATH_TEST_TWO = "dataset/claim_explanation_test_two_ID_reasoning.json"

# manually annotated chart annotation subset
DATASET_PATH_TEST = "dataset/test_one_annotated_charts.json"
DATASET_PATH_TEST_TWO = "dataset/test_two_annotated_charts.json"

PATH_DEPLOT_TABLES = "/scratch/users/k20116188/chart-fact-checking/deplot-tables"

# OUTPUT_DIR = "./results/chart_table_classification_DeBERTa"
OUTPUT_DIR = "./results/chart_table_classification_DeBERTa_CHARTATTRIBUTE_SUBSET"  # todo set output directory before running

SAVE_MODEL_PATH = "./results/chart_table_classification"
HG_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# HG_MODEL_NAME = "./results/chart_table_classification_DeBERTa"


LABEL_DICT = {
    "Yes": 0,
    "No": 1,
    "TRUE": 0,
    "FALSE": 1,
}

LABEL_DICT_REVERSE = {
    "0": "True",
    "1": "False",
    "2": "NEI",
}

BS = 2
NUM_EPOCHS = 3
METRIC = evaluate.load("glue", "mrpc")
TRAINING_ARGS = TrainingArguments(
    output_dir='./results/chart_table_classification',  # output directory
    num_train_epochs=NUM_EPOCHS,  # total number of training epochs
    per_device_train_batch_size=BS,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=50,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    eval_steps=300,
    save_steps=300,
    metric_for_best_model="eval_f1_micro",
    save_total_limit=1,
    load_best_model_at_end=True,
    learning_rate=1e-06,
    fp16=True,  # mixed precision training
)


def _read_deplot_table(image_name: str):
    path_table = os.path.join(PATH_DEPLOT_TABLES,
                              os.path.basename(image_name) + ".txt")

    with open(path_table, "r", encoding="utf-8") as f:
        return str(f.readlines())


def _read_chart_dataset(dataset):
    claims = []
    evidences = []
    labels = []

    for item in dataset:
        try:
            table = _read_deplot_table(item["chart_img"])
            claim = str(item["claim"])
            label = str(item["label"])
            caption = str(item["caption"])

            # table = "" # using for the hypothesis-only baseline
            # caption = "" # using for the hypothesis-only baseline
        except IndexError as e:
            print(f"Exception for file {item['chart_img']}: {e}.")
            continue

        claims.append(claim)
        evidences.append(table + ". " + caption)
        labels.append(LABEL_DICT[label])

    return claims, evidences, labels


def _tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def _compute_metrics(eval_preds):
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


def train(model, training_args, train_dataset, dev_dataset, test_dataset, save_path, only_test=False):
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=_compute_metrics,
    )
    if not only_test:
        trainer.train()
        trainer.save_model(save_path)

    result_dict = trainer.predict(test_dataset)
    print(result_dict.metrics)
    return result_dict


def continue_training(model, training_args, train_dataset, dev_dataset, test_dataset):
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=_compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(SAVE_MODEL_PATH)

    result_dict = trainer.predict(test_dataset)
    print(result_dict.metrics)
    return result_dict


def _load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _create_dataset(input_data):
    claims, tables, labels = _read_chart_dataset(input_data)
    data_tokenized = tokenizer(claims, tables,
                               max_length=MAX_LENGTH,
                               return_token_type_ids=True, truncation=True,
                               padding=True)
    return ChartDataset(data_tokenized, labels)


if __name__ == "__main__":
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(HG_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained("./results/chart_table_classification_DeBERTa",
                                                               torch_dtype="auto")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    # Dataset preparation
    train_data = _load_dataset(DATASET_PATH_TRAIN)
    test_data = _load_dataset(DATASET_PATH_TEST)
    test_two_data = _load_dataset(DATASET_PATH_TEST_TWO)
    val_data = _load_dataset(DATASET_PATH_VAL)

    train_dataset = _create_dataset(train_data)
    test_dataset = _create_dataset(test_data)
    test_two_dataset = _create_dataset(test_two_data)
    val_dataset = _create_dataset(val_data)

    # Check if the directory exists, otherwise create
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    results_dict = train(model, TRAINING_ARGS, train_dataset=train_dataset,
                         dev_dataset=val_dataset, test_dataset=test_dataset, only_test=True, save_path=OUTPUT_DIR)

    print(f"len(results_dict.predictions.tolist()): {len(results_dict.predictions.tolist())}")

    results_dict_two = train(model, TRAINING_ARGS, train_dataset=train_dataset,
                             dev_dataset=val_dataset, test_dataset=test_two_dataset, only_test=True,
                             save_path=OUTPUT_DIR)
    print(f"len(results_dict_two.predictions.tolist()): {len(results_dict_two.predictions.tolist())}")

    with open(os.path.join(OUTPUT_DIR, "output_test.txt"), "w") as f:
        f.write(f"result_dict.metrics: {results_dict.metrics}\n\n")

        for i, logits in enumerate(results_dict.predictions.tolist()):
            predictions = np.argmax(logits, axis=-1)
            if 'claim_id' in test_data[i].keys():
                f.write(f"claim_id: {test_data[i]['claim_id']}\n")
            f.write(f"input: {tokenizer.decode(test_dataset[i]['input_ids'])}\n")
            f.write(f"label: {LABEL_DICT_REVERSE[str(results_dict.label_ids.tolist()[i])]}\n")
            f.write(f"prediction: {LABEL_DICT_REVERSE[str(predictions)]}\n\n")

    with open(os.path.join(OUTPUT_DIR, "output_test_two.txt"), "w") as f:
        f.write(f"result_dict.metrics: {results_dict_two.metrics}\n\n")

        for i, logits in enumerate(results_dict_two.predictions.tolist()):
            predictions = np.argmax(logits, axis=-1)
            if 'claim_id' in test_two_data[i].keys():
                f.write(f"claim_id: {test_two_data[i]['claim_id']}\n")
            f.write(f"input: {tokenizer.decode(test_two_dataset[i]['input_ids'])}\n")
            f.write(f"label: {LABEL_DICT_REVERSE[str(results_dict_two.label_ids.tolist()[i])]}\n")
            f.write(f"prediction: {LABEL_DICT_REVERSE[str(predictions)]}\n\n")
