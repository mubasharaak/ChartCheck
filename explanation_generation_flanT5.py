import json
import os

import evaluate
import nltk
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from nltk.tokenize import sent_tokenize
nltk.download("punkt")


# variables
max_length = 1024
train_dataset_path = os.path.join("data", "")
test_dataset_path = os.path.join("data", "")
dev_dataset_path = os.path.join("data", "")
# metric = load_metric("rouge")
metric = evaluate.load("rouge")

DATASET_PATH = "claim_explanation_verification_pre_tasksets.json"

label_dict = {
    "Yes": 0,
    "No": 1,
    "TRUE": 0,
    "FALSE": 1,
}
label_dict_reverse = {
    "Yes": "true",
    "No": "false",
    "TRUE": "true",
    "FALSE": "false",
}


# functions
def join_unicode(delim, entries):
    return delim.join(entries)


max_input_length = 1024
max_target_length = 128


def read_table(chart_filename: str):
    path_table = os.path.join("/scratch/users/k20116188/chart-fact-checking/deplot-tables",
                              os.path.basename(chart_filename) + ".txt")
    with open(path_table, "r", encoding="utf-8") as f:
        table = str(f.readlines())
    return table


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def preprocess_function(examples):
    inputs = [f"Explain why '{doc['claim']}' as {label_dict_reverse[doc['label']]} given this table: {read_table(doc['chart_img'])}." for doc in examples]
    print(f"len(inputs): {len(inputs)}")

    # inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    labels = tokenizer(text_target=[doc["explanation"] for doc in examples], max_length=max_target_length, padding="max_length", truncation=True)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    print(f"model_inputs: {len(model_inputs['labels'])}")

    return model_inputs


class ChartDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["labels"])


training_args = Seq2SeqTrainingArguments(
    output_dir='./results/chart_explanation_FlanT5',  # output directory
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=5,
    # logging & evaluation strategies
    logging_dir="./results/chart_explanation_FlanT5/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_rougeL",
    push_to_hub=False,
)


def train(model, training_args, train_dataset, dev_dataset, test_dataset, only_test=False):
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    if not only_test:
        trainer.train()
        trainer.save_model("./results/chart_explanation_FlanT5")

    result_dict = trainer.predict(test_dataset)
    print(result_dict.metrics)

    return result_dict


if __name__ == "__main__":
    # Load model
    hg_model_hub_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(hg_model_hub_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    # load file
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    np.random.seed(42)
    data = np.array(data)

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
    train_input = preprocess_function(train_data)
    test_input = preprocess_function(test_data)
    val_input = preprocess_function(val_data)

    print(f"val input: {len(val_input)}")

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    train_dataset = ChartDataset(train_input)
    val_dataset = ChartDataset(val_input)
    test_dataset = ChartDataset(test_input)

    results_dict = train(model, training_args, train_dataset=train_dataset,
                         dev_dataset=val_dataset, test_dataset=test_dataset, only_test=False)

    with open("./results/chart_explanation_FlanT5/test_output.txt", "w") as f:
        json.dump(results_dict, f, indent=4)
