import json
import os
import re

import evaluate
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
# nltk.download("punkt")
from sklearn.metrics import f1_score
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer

# variables
MAX_LENGTH = 2048
MAX_INPUT_LEN = 1024
MAX_TARGET_LEN = 128
DATASET_PATH_TRAIN = "dataset/claim_explanation_verification_pre_tasksets_train_V2.json"
DATASET_PATH_VAL = "dataset/claim_explanation_verification_pre_tasksets_validation_V2.json"
DATASET_PATH_TEST = "dataset/claim_explanation_verification_pre_tasksets_test_V2.json"
DATASET_PATH_TEST_TWO = "dataset/claim_explanation_verification_pre_tasksets_test_two_V2.json"

FINETUNING = True
ONLY_TEST = True
FEWSHOT = False
ZEROSHOT = False
ONLY_EXPLANATION = False
ONLY_CLASSIFICATION = False
CLASSIFICATIONS_BY = ""

if FINETUNING and not ONLY_TEST:
    TOKENIZER_PATH = HG_MODEL_HUB_NAME = "google/flan-t5-base"
elif FINETUNING and ONLY_TEST:
    HG_MODEL_HUB_NAME = "/scratch/users/k20116188/chart-fact-checking/chart_classification_explanation_FlanT5_finetune/checkpoint-2200"
    TOKENIZER_PATH = "google/flan-t5-base"
else:
    # zero or few shot setting
    TOKENIZER_PATH = HG_MODEL_HUB_NAME = "google/flan-t5-xl"

LABEL_DICT = {
    "Yes": 0,
    "No": 1,
    "TRUE": 0,
    "FALSE": 1,
    "true": 0,
    "false": 1,
}
LABEL_DICT_REVERSE = {
    "Yes": "true",
    "No": "false",
    "TRUE": "true",
    "FALSE": "false",
}

INIT_PROMPT_FEW_SHOT = """
Given a claim, a table and it's caption, decide if the claim is true or false. 

Examples: 
Classify and explain if claim 'Coal is the second-lowest source of energy for electricity production in Romania.' is true or false given this caption: 'Electricity production in Romania by source of energy.' and this table: 'TITLE | \n Other | 2 \n Hydro | 36% \n Coal | 33% \n Nuclear | 19% \n Gas | 10%'.
Answer: false. The chart shows that Coal contributes to 33% of the electricity production in Romania, which is the second-highest percentage among all the sources of energy listed in the chart.

Classify and explain if claim 'Hydro is the primary source of energy for electricity production in Romania.' is true or false given this caption: 'Electricity production in Romania by source of energy.' and this table: 'TITLE | \n Other | 2 \n Hydro | 36% \n Coal | 33% \n Nuclear | 19% \n Gas | 10%'.
Answer: true. The chart shows that Hydro contributes to 36% of the electricity production in Romania, which is the highest percentage among all the sources of energy listed in the chart. 

Complete the following:
"""

INIT_PROMPT_ZERO_SHOT = """
Given a claim, a table and it's caption, decide if the claim is true or false. 

Complete the following:

"""


# functions
def join_unicode(delim, entries):
    return delim.join(entries)


def _read_table(chart_filename: str):
    path_table = os.path.join("/scratch/users/k20116188/chart-fact-checking/deplot-tables",
                              os.path.basename(chart_filename) + ".txt")
    with open(path_table, "r", encoding="utf-8") as f:
        table = str(f.readlines())
    return table


def _postprocess_text(preds, labels):
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
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = _postprocess_text(decoded_preds, decoded_labels)
    if not ONLY_EXPLANATION:
        class_label_preds = []
        class_label_gold = []
        for i, entry in enumerate(decoded_preds):
            if "true" in entry.lower():
                class_label_preds.append(LABEL_DICT["true"])
                gold_answer = decoded_labels[i].lower().split(".")[0]
                class_label_gold.append(LABEL_DICT[gold_answer])
            elif "false" in entry.lower():
                class_label_preds.append(LABEL_DICT["false"])
                gold_answer = decoded_labels[i].lower().split(".")[0]
                class_label_gold.append(LABEL_DICT[gold_answer])
            else:
                continue

    # class_label_preds = [label_dict[entry.lower().split("the claim is ")[1].split(".")[0]] for entry in decoded_preds]
    # class_label_gold = [label_dict[entry.lower().split("the claim is ")[1].split(".")[0]] for entry in decoded_labels]

    # rouge
    metric = evaluate.load("rouge")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    #
    # # bleu
    # metric = evaluate.load("bleu")
    # result_bleu = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # result.update(result_bleu)
    #
    # # bertscore
    # metric = evaluate.load("bertscore")
    # result_bertscore = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    # result_bertscore = {k: sum(v) / len(v) for k, v in result_bertscore.items() if k in ['precision', 'recall', 'f1']}
    # result_bertscore["bertscore_precision"] = result_bertscore.pop("precision")
    # result_bertscore["bertscore_recall"] = result_bertscore.pop("recall")
    # result_bertscore["bertscore_f1"] = result_bertscore.pop("f1")
    # result.update(result_bertscore)
    #
    # # meteor
    # metric = evaluate.load("meteor")
    # result_meteor = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # result.update(result_meteor)
    #
    # # bleurt
    # metric = evaluate.load('bleurt')
    # result_bleurt = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # result_bleurt = {k: sum(v) / len(v) for k, v in result_bleurt.items()}
    # result_bleurt["bleurt"] = result_bleurt.pop("scores")
    # result.update(result_bleurt)

    if not ONLY_EXPLANATION:
        # classification results
        f1_micro = f1_score(y_true=class_label_gold, y_pred=class_label_preds, average='micro')
        f1_macro = f1_score(y_true=class_label_gold, y_pred=class_label_preds, average='macro')
        result["f1_micro"] = f1_micro
        result["f1_macro"] = f1_macro

    # add predictions
    result['preds'] = decoded_preds

    return result


def preprocess_claim_for_pred_label(claim):
    return re.sub(r'\s+([.,])', r'\1', re.sub(' +', ' ', str(claim).strip()))


def preprocess_function_explanation(examples, is_testset=False, is_testset_two=False):
    if CLASSIFICATIONS_BY.lower() == "deberta" and (is_testset or is_testset_two):
        if is_testset:
            inputs = [

                f"Explain why '{doc['claim']}' is {dict_test_pred[preprocess_claim_for_pred_label(doc['claim'])]} given this caption: {doc['caption']} and table: {_read_table(doc['chart_img'])}."
                for doc in examples]

        else:  # is_testset_two
            inputs = [
                f"Explain why '{doc['claim']}' is {dict_test_two_pred[preprocess_claim_for_pred_label(doc['claim'])]} given this caption: {doc['caption']} and table: {_read_table(doc['chart_img'])}."
                for doc in examples]
    else:  # training or val set
        inputs = [
            f"Explain why '{doc['claim']}' is {LABEL_DICT_REVERSE[doc['label']]} given this caption: {doc['caption']} and table: {_read_table(doc['chart_img'])}."
            for doc in examples]

    # inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LEN, padding="max_length", truncation=True)

    labels = tokenizer(text_target=[doc["explanation"] for doc in examples], max_length=MAX_TARGET_LEN,
                       padding="max_length", truncation=True)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def preprocess_function_classification_explanation(examples, training_setting="finetune"):
    if training_setting == "finetune":
        inputs = [
            f"Classify and explain if claim '{doc['claim']}' is true or false given this caption: {doc['caption']} and this table: {_read_table(doc['chart_img'])}.\n Answer:"
            for doc in examples]
    elif training_setting == "fewshot":
        inputs = [
            f"{INIT_PROMPT_FEW_SHOT} Classify and explain if claim '{doc['claim']}' is true or false given this caption: {doc['caption']} and this table: {_read_table(doc['chart_img'])}.\n Answer:"
            for doc in examples]
    elif training_setting == "zeroshot":
        inputs = [
            f"{INIT_PROMPT_ZERO_SHOT} Classify and explain if claim '{doc['claim']}' is true or false given this caption: {doc['caption']} and this table: {_read_table(doc['chart_img'])}.\n Answer:"
            for doc in examples]

    # inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LEN, padding="max_length", truncation=True)

    # Input instructioon: "{label}. {explanation}"
    print("First example: {}".format(examples[0]))
    labels = tokenizer(text_target=["{}. {}".format(LABEL_DICT_REVERSE[doc['label']], doc["explanation"]) for doc in examples],
                       max_length=MAX_TARGET_LEN, padding="max_length", truncation=True)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def preprocess_function_classification(examples, training_setting="finetune"):
    if training_setting == "finetune":
        inputs = [
            f"Classify if claim '{doc['claim']}' is true or false given this caption: {doc['caption']} and this table: {_read_table(doc['chart_img'])}.\n Answer:"
            for doc in examples]
    elif training_setting == "fewshot":
        inputs = [
            f"{INIT_PROMPT_FEW_SHOT} Classify if claim '{doc['claim']}' is true or false given this caption: {doc['caption']} and this table: {_read_table(doc['chart_img'])}.\n Answer:"
            for doc in examples]
    elif training_setting == "zeroshot":
        inputs = [
            f"{INIT_PROMPT_ZERO_SHOT} Classify if claim '{doc['claim']}' is true or false given this caption: {doc['caption']} and this table: {_read_table(doc['chart_img'])}.\n Answer:"
            for doc in examples]

    # inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LEN, padding="max_length", truncation=True)

    # Input instructioon: "{label}. {explanation}"
    labels = tokenizer(text_target=[f"{LABEL_DICT_REVERSE[doc['label']]}. " for doc in examples],
                       max_length=MAX_TARGET_LEN, padding="max_length", truncation=True)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


class ChartDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["labels"])


def train(model, training_args, train_dataset, dev_dataset, test_dataset, save_path, only_test=False):
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
        trainer.save_model(save_path)

    result_dict = trainer.predict(test_dataset)
    print(result_dict.metrics)

    return result_dict


def _load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(HG_MODEL_HUB_NAME)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    if FINETUNING and not ONLY_TEST:
        model.train()

    # load files (train, validation, first and second test set)
    train_data = _load_dataset(DATASET_PATH_TRAIN)
    val_data = _load_dataset(DATASET_PATH_VAL)
    test_data = _load_dataset(DATASET_PATH_TEST)
    test_two_data = _load_dataset(DATASET_PATH_TEST_TWO)

    # load and replace labels for DeBERTa/TAPAS classification for testsets
    if CLASSIFICATIONS_BY.lower() == "deberta":
        with open(
                "/users/k20116188/projects/chartcheck/ChartFC/results/chart_table_classification_DeBERTa/output_test.txt",
                "r", encoding="utf-8") as file:
            test_predictions = file.readlines()
        dict_test_pred = {}
        for i, entry in enumerate(test_predictions):
            if entry.startswith("input"):
                claim = entry.split("[CLS]")[1].split("[SEP]")[0].strip()
                claim = re.sub(' +', ' ', claim)
                if "false" in test_predictions[i + 2].lower():
                    pred_label = "False"
                else:
                    pred_label = "True"
                dict_test_pred[str(claim)] = pred_label

        with open(
                "/users/k20116188/projects/chartcheck/ChartFC/results/chart_table_classification_DeBERTa/output_test_two.txt",
                "r", encoding="utf-8") as file:
            test_two_predictions = file.readlines()
        dict_test_two_pred = {}
        for i, entry in enumerate(test_two_predictions):
            if entry.startswith("input"):
                claim = entry.split("[CLS]")[1].split("[SEP]")[0].strip()
                claim = re.sub(' +', ' ', claim)
                if "false" in test_two_predictions[i + 2].lower():
                    pred_label = "False"
                else:
                    pred_label = "True"
                dict_test_two_pred[str(claim)] = pred_label

    # Dataset preperation
    if ONLY_EXPLANATION:
        train_input = preprocess_function_explanation(train_data)
        val_input = preprocess_function_explanation(val_data)
        test_input = preprocess_function_explanation(test_data, is_testset=True)
        test_two_input = preprocess_function_explanation(test_two_data, is_testset_two=True)
    elif ONLY_CLASSIFICATION:
        train_input = preprocess_function_classification(train_data)
        val_input = preprocess_function_classification(val_data)
        test_input = preprocess_function_classification(test_data)
        test_two_input = preprocess_function_classification(test_two_data)
    else:
        if FINETUNING:
            train_input = preprocess_function_classification_explanation(train_data, "finetune")
            test_input = preprocess_function_classification_explanation(test_data, "finetune")
            test_two_input = preprocess_function_classification_explanation(test_two_data, "finetune")
            val_input = preprocess_function_classification_explanation(val_data, "finetune")
        elif FEWSHOT:
            train_input = preprocess_function_classification_explanation(train_data, "fewshot")
            test_input = preprocess_function_classification_explanation(test_data, "fewshot")
            test_two_input = preprocess_function_classification_explanation(test_two_data, "fewshot")
            val_input = preprocess_function_classification_explanation(val_data, "fewshot")
        elif ZEROSHOT:
            train_input = preprocess_function_classification_explanation(train_data, "zeroshot")
            test_input = preprocess_function_classification_explanation(test_data, "zeroshot")
            test_two_input = preprocess_function_classification_explanation(test_two_data, "zeroshot")
            val_input = preprocess_function_classification_explanation(val_data, "zeroshot")
        else:
            print("Error: Training setting is missing!")

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
    test_two_dataset = ChartDataset(test_two_input)

    task = "explanation_only" if ONLY_EXPLANATION else "classification_explanation"
    if FINETUNING:
        if CLASSIFICATIONS_BY.lower() == "deberta":
            output_path = f"/scratch/users/k20116188/chart-fact-checking/chart_{task}_FlanT5_finetune_deberta_classification"
        elif ONLY_CLASSIFICATION:
            output_path = f"/scratch/users/k20116188/chart-fact-checking/chart_{task}_FlanT5_finetune_only_classification"
        else:
            output_path = f"/scratch/users/k20116188/chart-fact-checking/chart_{task}_FlanT5_finetune"
    elif FEWSHOT:
        output_path = f"/scratch/users/k20116188/chart-fact-checking/chart_{task}_FlanT5_fewshot_2shots"
    elif ZEROSHOT:
        output_path = f"/scratch/users/k20116188/chart-fact-checking/chart_{task}_FlanT5_zeroshot"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    path_testset_one_metrics = os.path.join(output_path, "metrics_testset_one.txt")
    path_testset_two_metrics = os.path.join(output_path, "metrics_testset_two.txt")
    path_testset_one_predictions = os.path.join(output_path, "predictions_testset_one.txt")
    path_testset_two_predictions = os.path.join(output_path, "predictions_testset_two.txt")
    path_testset_one_df = os.path.join(output_path, "predictions_testset_one.csv")
    path_testset_two_df = os.path.join(output_path, "predictions_testset_two.csv")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,  # output directory
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        fp16=False,  # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=10,
        logging_strategy="steps",
        logging_steps=100,
        logging_dir=os.path.join(output_path, "logs"),
        eval_steps=100,
        save_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
        generation_max_length=100
    )

    if FINETUNING:
        results_dict = train(model, training_args, train_dataset=train_dataset,
                             dev_dataset=val_dataset, test_dataset=test_dataset, save_path=output_path,
                             only_test=ONLY_TEST)
        results_dict_two = train(model, training_args, train_dataset=train_dataset,
                                 dev_dataset=val_dataset, test_dataset=test_two_dataset, save_path=output_path,
                                 only_test=ONLY_TEST)
    else:
        results_dict = train(model, training_args, train_dataset=train_dataset,
                             dev_dataset=val_dataset, test_dataset=test_dataset, save_path=output_path, only_test=True)
        results_dict_two = train(model, training_args, train_dataset=train_dataset,
                                 dev_dataset=val_dataset, test_dataset=test_two_dataset, save_path=output_path,
                                 only_test=True)

    with open(path_testset_one_metrics, "w") as f:
        f.write(f"result_dict.metrics: {results_dict.metrics}\n\n")

    with open(path_testset_two_metrics, "w") as f:
        f.write(f"result_dict.metrics: {results_dict_two.metrics}\n\n")

    for results, path, dataset, path_df in zip([results_dict, results_dict_two],
                                      [path_testset_one_predictions, path_testset_two_predictions],
                                      [test_dataset, test_two_dataset], [path_testset_one_df, path_testset_two_df]):
        claims = []
        preds = []
        with open(os.path.join(output_path, path), "w") as f:
            for i, decoded_pred in enumerate(results.metrics['test_preds']):
                input_text = tokenizer.decode(dataset[i]['input_ids']).split('Answer')[0]
                f.write("input: {}\n".format(input_text))
                f.write("prediction: {}\n\n".format(decoded_pred))

                claims.append(input_text)
                preds.append(decoded_pred.split("\n")[1])

        results_df = pd.DataFrame({'claims': claims, 'predicted_explanation': results.metrics['test_preds']})
        results_df.to_csv(path_df)
