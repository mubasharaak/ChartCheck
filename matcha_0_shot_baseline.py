import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import json
from PIL import Image
import os
import torch
from itertools import cycle
from evaluate import load
from tqdm import tqdm
import pickle

model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-chartqa")
processor = AutoProcessor.from_pretrained("google/matcha-chartqa")
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("claim_explanation_verification_pre_tasksets.json", "r") as f:
    data = json.load(f)

for example in data:
    example["label"] = "Yes" if example["label"] == "TRUE" else "No"

new_data = []
for example in data:
    try:
        imgname = os.path.basename(example["chart_img"])
        Image.open(f"ChartFC/{imgname}").convert('RGB')
        new_data.append(example)
    except Exception as e:
        print(e)
        break
        pass

data = np.array(new_data)

np.random.seed(42)

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

len(train_data)

MAX_PATCHES = 2048


class ChartFCDataset(Dataset):
    def __init__(self, processor, root_dir="ChartFC", split='train'):
        """
        Args:
            root_dir (string): Directory with all the ChartQA data.
            split (string): Which split to load ("train" or "val" or "test").
            split2 (string): Which split to load ("both" or "augmented" or "human") within the first split.
        """
        self.processor = processor
        self.root_dir = root_dir
        self.split = split
        self.image_dir = root_dir

        if split == 'train':
            self.data = train_data
        elif split == "val":
            self.data = val_data
        elif split == "test":
            self.data = test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qa = self.data[idx]
        # Load image
        imgname = os.path.basename(qa["chart_img"])
        qa["imgname"] = imgname
        qa["image"] = Image.open(f"{self.image_dir}/{imgname}").convert('RGB')
        return qa


question = f"is the following claim supported by the chart:"


def collator(batch):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    images = [item["image"] for item in batch]
    header_texts = [f"{question} {item['claim']} (Yes/No)" for item in batch]
    label_texts = [f"{item['label']}" for item in batch]  # because {item['explanation']}

    inputs = processor(images=images, text=header_texts, return_tensors="pt")
    labels = processor.tokenizer(label_texts, padding="max_length", return_tensors="pt", max_length=256)
    new_batch["labels"] = labels.input_ids
    new_batch["flattened_patches"] = inputs["flattened_patches"]
    new_batch["attention_mask"] = inputs["attention_mask"]
    new_batch["header_texts"] = header_texts
    new_batch["imgname"] = [item["imgname"] for item in batch]

    return new_batch


batch_size = 1

train_dataset = ChartFCDataset(processor, split='train')
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator)

model.to(device);

exact_match_metric = load("exact_match")

model.eval()

accuracy = []
predictions = []
for idx, batch in tqdm(enumerate(train_dataloader)):
    labels = batch.pop("labels").to(device)
    flattened_patches = batch.pop("flattened_patches").to(device)
    attention_mask = batch.pop("attention_mask").to(device)
    imgnames = batch.pop("imgname")
    header_texts = batch.pop("header_texts")

    generated_ids = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=128)
    predicted_answer = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    for img, header, generated_id, answer, label in zip(imgnames, header_texts, generated_ids, predicted_answer,
                                                        labels):
        predictions.append(
            {"imgname": imgname, "query": header, "generated_ids": generated_id.cpu().numpy(), "decoded_answer": answer,
             "label": label})

metric = exact_match_metric.compute(predictions=[item["decoded_answer"] for item in predictions],
                                    references=[item["label"] for item in predictions])

print(metric)

with open('accuracy_chartqa_new_prompt.pkl', 'wb') as f:
    pickle.dump(accuracy, f)

# files.download('accuracy_chartqa_new_prompt.pkl')

with open('predictions_chartqa_new_prompt.pkl', 'wb') as f:
    pickle.dump(predictions, f)

# files.download('predictions_chartqa_new_prompt.pkl')

### Loading Pickle for analysis

with open("accuracy_plotqav2.pkl", "rb") as f:
    a = pickle.load(f)

with open("predictions_plotqav2.pkl", "rb") as f:
    predictions = pickle.load(f)

sum(accuracy) / len(accuracy)

sum([int(item['label'] == item['decoded_answer']) for item in predictions]) / len(predictions)

len(set([item['imgname'] for item in predictions]))

with open("barchart_horizontal.json", "r") as f:
    bar_horizontal = json.load(f)[0]

bar_horizontal = set([os.path.splitext(i["file_name"])[0] for i in bar_horizontal])

with open("barchart_vertical.json", "r") as f:
    bar_vertical = json.load(f)[0]

bar_vertical = set([os.path.splitext(i["file_name"])[0] for i in bar_vertical])

with open("line_chart.json", "r") as f:
    line_chart = json.load(f)[0]

line_chart = set([os.path.splitext(i["file_name"])[0] for i in line_chart])

with open("pie_chart.json", "r") as f:
    pie_chart = json.load(f)[0]

pie_chart = set([os.path.splitext(i["file_name"])[0] for i in pie_chart])

for item in predictions:
    filename = os.path.splitext(item['imgname'])[0]
    key = filename
    set1 = bar_horizontal
    set2 = bar_vertical
    set3 = line_chart
    set4 = pie_chart
    if key in set1 and key not in set2 and key not in set3 and key not in set4:
        item["chart_type"] = "bar_horizontal"
    elif key not in set1 and key in set2 and key not in set3 and key not in set4:
        item["chart_type"] = "bar_vertical"
    elif key not in set1 and key not in set2 and key in set3 and key not in set4:
        item["chart_type"] = "line_chart"
    elif key not in set1 and key not in set2 and key not in set3 and key in set4:
        item["chart_type"] = "pie_chart"
    else:
        item["chart_type"] = "mixed"

accuracy_by_category = {"bar_horizontal": [], "bar_vertical": [], "line_chart": [], "pie_chart": [], "mixed": []}

for item in predictions:
    metric = int(item["decoded_answer"] == item["label"])
    accuracy_by_category[item["chart_type"]].append(metric)

