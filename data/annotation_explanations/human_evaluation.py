import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score
import seaborn as sns

_MANUAL_EVAL_SUBSET_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/data/annotation_explanations/claim_explanation_verification_pre_tasksets_test_V2_annotation_subset.csv"
_MANUAL_EVAL_SUBSET_ANNOTATED_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/data/annotation_explanations/claim_explanation_verification_pre_tasksets_test_V2_annotation_subset_annotated.csv"
_LABEL_MAPPING = {
    'FALSE': 0,
    'TRUE': 1,
    'False': 0,
    'True': 1,
    False: 0,
    True: 1,
    "Its not 'True' neither 'False'": 2,
}

# load each file
input = pd.read_csv(_MANUAL_EVAL_SUBSET_PATH)
annotated = pd.read_csv(_MANUAL_EVAL_SUBSET_ANNOTATED_PATH)

# human performance
preds = []
targets = []
for _, row in annotated.iterrows():
    if _LABEL_MAPPING[row['label']] == 2:
        print("continue...")
        continue
    index = row['ID']
    preds.append(_LABEL_MAPPING[row['label']])
    targets.append(_LABEL_MAPPING[input.iloc[index]['label']])

# calculate accuracy and F1
eval_scores = {"f1_micro": f1_score(y_true=targets, y_pred=preds, average='micro'),
               "f1_macro": f1_score(y_true=targets, y_pred=preds, average='macro'),
               "accuracy": accuracy_score(y_true=targets, y_pred=preds),
               # "recall": recall_score(y_true=targets, y_pred=preds)
               }
print(eval_scores)

#
sns.barplot(annotated["factuality"])
