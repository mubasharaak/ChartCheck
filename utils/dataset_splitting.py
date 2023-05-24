import json
import numpy as np
from random import shuffle


path = r"/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/claim_explanation_verification_pre_tasksets.json"

with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

charts = [entry["chart_img"] for entry in data]
unique_charts = list(set(charts))

# print(f"Number of charts: {len(charts)}")
# print(f"Number of unique charts: {len(unique_charts)}")

np.random.seed(42)
unique_charts = np.array(unique_charts)
# Shuffle the indices of the data
indices_charts = np.random.permutation(len(unique_charts))
num_test_charts = int(0.09 * len(unique_charts))

# Split the indices into training, validation, and testing sets
test_two_indices = indices_charts[:num_test_charts]
test_two_charts = unique_charts[test_two_indices]

other_data = []
train_data = []
test_data = []
val_data = []
test_data_two = []

print(f"Number of unique charts in testset two: {len(test_two_charts)}")

counter = 0
for entry in data:
    if entry["chart_img"] in test_two_charts:
        test_data_two.append(entry)
        counter += 1
    else:
        other_data.append(entry)

# randomly permutate other_data
shuffle(other_data)

# split other data in train, test, val => 80/10/10
len_data = len(other_data)
print(f"len_data: {len_data}")
train_data = other_data[:int(len_data*0.8)]
val_data = other_data[int(len_data*0.8):int(len_data*0.9)]
test_data = other_data[int(len_data*0.9):]


print("Summary")
print(f"train_data: {len(train_data)}")
print(f"val_data: {len(val_data)}")
print(f"test_data: {len(test_data)}")
print(f"test_data_two: {len(test_data_two)}")

path_train = r"/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/claim_explanation_verification_pre_tasksets_train.json"
path_val = r"/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/claim_explanation_verification_pre_tasksets_validation.json"
path_test = r"/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/claim_explanation_verification_pre_tasksets_test.json"
path_test_two = r"/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/ChartFC/claim_explanation_verification_pre_tasksets_test_two.json"

with open(path_train, "w", encoding="utf-8") as file:
    json.dump(train_data, file, indent=4)

with open(path_val, "w", encoding="utf-8") as file:
    json.dump(val_data, file, indent=4)

with open(path_test, "w", encoding="utf-8") as file:
    json.dump(test_data, file, indent=4)

with open(path_test_two, "w", encoding="utf-8") as file:
    json.dump(test_data_two, file, indent=4)

# check to make sure test_data_two charts dont occure in other dataset splits
test_two_charts = [sample["chart_img"] for sample in test_data_two]
for train_sample in test_data_two:
    if train_sample["chart_img"] in test_two_charts:
        print(f"double entry!")

print("Done!")
