import argparse
import json
import os

import openai

import openai_classification_explanation
import random

parser = argparse.ArgumentParser(
    description='Prompt Scorer OpenAI arguments'
)
parser.add_argument(
    '--test_set_path',
    default="dataset/claim_explanation_verification_pre_tasksets_test_V2.json",
    help='Path to testdata.'
)
parser.add_argument(
    '--test_set_two_path',
    default="dataset/claim_explanation_verification_pre_tasksets_test_two_V2.json",
    help='Path to testdata two.'
)
parser.add_argument(
    '--output_path',
    default="chart_table_classification_{}_classify_n_explain_zeroshot_{}",
    help='Path to output file for scores.'
)
parser.add_argument(
    '--only_evaluate_no_prediction',
    default=False,
    action="store_true",
    help='Given predictions_output_path load predictions for evaluation.'
)
parser.add_argument(
    '--zero_shot',
    default=False,
    action="store_true",
    help='Zero or few shot prompting.'
)
parser.add_argument(
    '--model',
    default="GPT4V",
    help='Model'
)

args = parser.parse_args()
_TEST_SET_PATH = args.test_set_path
_TEST_SET_TWO_PATH = args.test_set_path
_ONLY_EVALUATE = args.only_evaluate_no_prediction
_ZERO_SHOT_EVAL = args.zero_shot
_MODEL = args.model

_OUTPUT_PATH = os.path.join("results", args.output_path.format(_MODEL, _ZERO_SHOT_EVAL))
if not os.path.exists(_OUTPUT_PATH):
    os.makedirs(_OUTPUT_PATH)

print("Results will be saved in {}".format(_OUTPUT_PATH))
_DEPLOT_TABLES_PATH = "/scratch/users/k20116188/chart-fact-checking/deplot-tables"
_KEY = open('openai_key.txt', 'r').read()
_CLIENT = openai.OpenAI(
    api_key=_KEY,
    timeout=10,
)
_RAND_SUBSET = 100
random.seed(10)


def _read_table(chart_filename: str):
    path_table = os.path.join(_DEPLOT_TABLES_PATH,
                              os.path.basename(chart_filename) + ".txt")
    with open(path_table, "r", encoding="utf-8") as f:
        table = str(f.readlines())
    return table


def _load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_jsonl_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def _load_jsonl_file(file_path):
    content = []
    with open(file_path, "r", encoding="utf-8") as f:
        for entry in f.readlines():
            content.append(json.loads(entry))
    return content


def main():
    index_mapping = {0: "one", 1: "two"}
    for i, path in enumerate([_TEST_SET_PATH, _TEST_SET_TWO_PATH]):
        _OUTPUT_PATH_PREDICTIONS = os.path.join(_OUTPUT_PATH, "predictions_testset_{}.jsonl".format(index_mapping[i]))
        _OUTPUT_PATH_METRICS = os.path.join(_OUTPUT_PATH, "metrics_testset_{}.json".format(index_mapping[i]))

        if _ONLY_EVALUATE:
            # Given predictions_output_path load predictions for evaluation
            predictions = _load_jsonl_file(os.path.join(_OUTPUT_PATH, ""))
        else:
            # predict using OpenAI API and evaluate
            input_data = _load_dataset(path)
            # select subset
            input_data = random.sample(input_data, _RAND_SUBSET)
            predictions = openai_classification_explanation.prompt_openai_model(input_data, _CLIENT, _ZERO_SHOT_EVAL)
            _save_jsonl_file(predictions, _OUTPUT_PATH_PREDICTIONS)

        scores = openai_classification_explanation.evaluate_openai_output(predictions)
        print("Output for test set {}: {}".format(index_mapping[i], scores))
        with open(_OUTPUT_PATH_METRICS, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)


if __name__ == '__main__':
    main()
