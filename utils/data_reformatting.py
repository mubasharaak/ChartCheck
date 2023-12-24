import pandas as pd

INPUT_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/data/claim_explanation_verification_pre_tasksets_train_V2.json"
OUTPUT_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/data/claim_explanation_verification_pre_tasksets_train_V2.csv"

df = pd.read_json(INPUT_PATH)
df.drop(["db_id", "claim_rewritten", "explanation_rewritten", "claim_error_corrected", "explanation_error_corrected",
         "label_claim", "label_explanation"], axis=1, inplace=True)

df.to_csv(OUTPUT_PATH)
