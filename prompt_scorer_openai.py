import os
import time
from dataclasses import dataclass

import openai
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score

_SEED = 10
_MODEL = "gpt-3.5-turbo-1106"
_MAX_TOKENS = 1500

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


@dataclass
class OpenAIResponse:
    id: int
    claim: str
    caption: str
    table: str
    label: str
    explanation: str
    response: str


_PROMPT = """
You will get a claim, a table and its caption. 
Decide if the claim is 'true' or 'false' based on the provided table and caption, explain your decision.
Use no further background knowledge but your commonsense to understand the table. 

-----
Examples: 
Claim: Coal is the second-lowest source of energy for electricity production in Romania.
Table: TITLE | \n Other | 2 \n Hydro | 36% \n Coal | 33% \n Nuclear | 19% \n Gas | 10%.
Caption: Electricity production in Romania by source of energy.
Answer: false. The chart shows that Coal contributes to 33% of the electricity production in Romania, which is the second-highest percentage among all the sources of energy listed in the chart.

Claim: Hydro is the primary source of energy for electricity production in Romania.
Table: TITLE | \n Other | 2 \n Hydro | 36% \n Coal | 33% \n Nuclear | 19% \n Gas | 10%.
Caption: Electricity production in Romania by source of energy.
Answer: true. The chart shows that Hydro contributes to 36% of the electricity production in Romania, which is the highest percentage among all the sources of energy listed in the chart. 

-----
Complete the following:
Claim: {}
Caption: {}
Table: {}
Answer:"""


def _query_openai(prompt: str, client, seed=_SEED, model=_MODEL, max_tokens=_MAX_TOKENS):
    return client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        max_tokens=max_tokens,
        # response_format={"type": response_format},
        seed=seed,
    )


def _get_response_text(response: openai.types.chat.chat_completion.ChatCompletion):
    return response.choices[0].message.content


def _process_output(dataset_sample,
                    response: openai.types.chat.chat_completion.ChatCompletion):
    dataset_sample["response"] = _get_response_text(response)
    return dataset_sample


def _read_table(chart_filename: str):
    path_table = os.path.join("/scratch/users/k20116188/chart-fact-checking/deplot-tables",
                              os.path.basename(chart_filename) + ".txt")
    with open(path_table, "r", encoding="utf-8") as f:
        table = str(f.readlines())
    return table


def _prepare_prompt(dataset_sample):
    """Formats prompt using dataset sample as input."""
    return _PROMPT.format(dataset_sample["claim"], _read_table(dataset_sample["chart_img"]), dataset_sample["caption"])


def prompt_openai_model(dataset: list, client):
    """Prompts OpenAI models."""
    responses = []
    for sample in dataset:
        # try:
        print("next sample.")
        prompt = _prepare_prompt(sample)
        while True:
            try:
                responses.append(_process_output(sample, _query_openai(prompt, client)))
                break
            except openai.APITimeoutError as e:
                print(e)
                time.sleep(10)
                pass
        # except Exception as e:
        #     print(e)
        #     continue
    return responses


def calculate_atomic_score(response: dict):
    try:
        if ("refute" in response and response["refute"] > 0) or (
                "contradicts" in response and response["contradicts"] > 0):
            # evidence clearly contradicts a sub-fact of the claim
            return 1
        elif "supports" in response:
            return response["supports"] / (response["supports"] + response["not enough information"])
        else:
            return response["support"] / (response["support"] + response["not enough information"])
    except Exception as e:
        return 0


def _postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds][1:]  # exclude first sentence containing the label
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def evaluate_openai_output(output):
    class_label_preds = []
    class_label_gold = []
    for i, entry in enumerate(output):
        response = entry["response"]
        if "true" in response.lower():
            class_label_preds.append(LABEL_DICT["true"])
            class_label_gold.append(LABEL_DICT[entry["label"]])
        elif "false" in response.lower():
            class_label_preds.append(LABEL_DICT["false"])
            class_label_gold.append(LABEL_DICT[entry["label"]])
        else:
            continue

    return {"f1_micro": f1_score(y_true=class_label_gold, y_pred=class_label_preds, average='micro'),
            "f1_macro": f1_score(y_true=class_label_gold, y_pred=class_label_preds, average='macro')}
