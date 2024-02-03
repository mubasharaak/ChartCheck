import json

import pandas as pd
import copy

_SAMPLE_ENTRY = {
    "id": 0,
    "chart_img": None,
    "caption": None,
    "label": None,
    "claim": None,
    "explanation": None
}

# subsample for explanation and human baseline annotation
path_excel = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/data/testset_reannotation/claim_explanation_verification_pre_tasksets_test_two_V2.xlsx"
path_csv = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/data/testset_reannotation/claim_explanation_verification_pre_tasksets_test_two_V2.csv"
path_json = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/data/testset_reannotation/claim_explanation_verification_pre_tasksets_test_two_V2.json"

dataset = pd.read_excel(path_excel, index_col=0)

# convert and save as csv
dataset.to_csv(path_csv)

# convert and save as json
samples = []
for index, row in dataset.iterrows():
    sample = copy.deepcopy(_SAMPLE_ENTRY)
    sample['id'] = index
    sample['chart_img'] = row['chart_img']
    sample['caption'] = row['caption']
    sample['label'] = "TRUE" if row['label'] == True else "FALSE"
    sample['claim'] = row['claim']
    sample['explanation'] = row['explanation']
    samples.append(sample)

with open(path_json, "w", encoding="utf-8") as file:
    json.dump(samples, file, indent=4)
