import json

import pandas as pd

INPUT_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/data/testset_reannotation/claim_explanation_verification_pre_tasksets_test_two_V2.csv"
OUTPUT_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/data/testset_reannotation/claim_explanation_verification_pre_tasksets_test_two_V2.json"

if INPUT_PATH.endswith(".json"):
    df = pd.read_json(INPUT_PATH)
    df.drop(["db_id", "claim_rewritten", "explanation_rewritten", "claim_error_corrected", "explanation_error_corrected",
             "label_claim", "label_explanation"], axis=1, inplace=True)

    df.to_csv(OUTPUT_PATH)
else:
    # takes csv as input and converts to json
    df = pd.read_csv(INPUT_PATH)
    df = df.rename(columns={"Unnamed: 0": "id"})
    df = df[["id", "chart_img", "caption", "label", "claim", "explanation"]]
    dicts = df.to_dict(orient='records')
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(dicts, file, indent=4)
