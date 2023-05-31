import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import json
from PIL import Image, UnidentifiedImageError
import cv2
import os
import torch
from itertools import cycle
from evaluate import load
from tqdm import tqdm
import pickle
from sklearn.metrics import f1_score
from transformers.models.pix2struct.image_processing_pix2struct import render_text

model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-base")
processor = AutoProcessor.from_pretrained("google/matcha-base")
device = "cuda:1" if torch.cuda.is_available() else "cpu"

def convert_RGBA(img):
    # Create a new image with a white background
    rgb_image = Image.new("RGB", img.size, (255, 255, 255))

    # Paste the RGBA image onto the RGB image
    rgb_image.paste(img, mask=img.split()[3])

    return rgb_image

class ChartFCDataset(Dataset):
    def __init__(self, processor, root_dir="ChartFC", split='train', split2="both", convert_image=True):
        """
        Args:
            root_dir (string): Directory with all the ChartQA data.
            split (string): Which split to load ("train" or "val" or "test").
        """
        self.processor = processor
        self.root_dir = root_dir
        self.split = split
        self.image_dir = root_dir
        self.convert_image = convert_image

        if split == 'train':
          with open("claim_explanation_verification_pre_tasksets_train.json", "r") as f:
            self.data = json.load(f)
        elif split=="val":
          with open("claim_explanation_verification_pre_tasksets_validation.json", "r") as f:
            self.data = json.load(f)
        elif split == "test":
          if split2 == "both":
            with open("claim_explanation_verification_pre_tasksets_test.json", "r") as f:
              seen= json.load(f)
            with open("claim_explanation_verification_pre_tasksets_test_two.json", "r") as f:
              unseen = json.load(f)
            self.data = seen + unseen
          elif split2 == "seen":
            with open("claim_explanation_verification_pre_tasksets_test.json", "r") as f:
              self.data = json.load(f)
          elif split2 == "unseen":
            with open("claim_explanation_verification_pre_tasksets_test_two.json", "r") as f:
              self.data = json.load(f)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        qa = self.data[idx]
        random = np.random.choice(self.data)
        # Load image
        imgname = os.path.basename(random["chart_img"])
        qa["imgname"] = imgname
        qa["caption"] = random["caption"]
        img = Image.open(f"{self.image_dir}/{imgname}")
        if self.convert_image and img.mode == "RGBA":
          qa["image"] = convert_RGBA(img)
        else:
          qa["image"] = img.convert("RGB")
        return qa

def get_final_label(label, expl):
  label = "Yes" if label == "TRUE" else "No"
  return f"{label}. Explanation: {expl}"

def get_decoder_prompt():
    return ""


def collator(batch):
  new_batch = {"flattened_patches":[], "attention_mask":[]}
  images = [item["image"] for item in batch]
  header_texts = [f"The chart below has the following caption: {item['caption']}. Given the caption and the chart below, is the following claim true: {item['claim']}? Explain why?" for item in batch]
  label_texts = [get_final_label(item['label'],item['explanation']) for item in batch]
  inputs = processor(images=images, text=header_texts, return_tensors="pt")
  decoder_inputs = processor.tokenizer(get_decoder_prompt(), return_tensors="pt", padding=True)
  labels = processor.tokenizer(label_texts, return_tensors="pt", padding=True)
  new_batch["labels"] = labels.input_ids
  new_batch["flattened_patches"] = inputs["flattened_patches"]
  new_batch["attention_mask"] = inputs["attention_mask"]
  new_batch["header_texts"] = header_texts
  new_batch["imgname"] = [item["imgname"] for item in batch]
  new_batch["decoder_inputs"] = decoder_inputs.input_ids
  new_batch["decoder_mask"] = decoder_inputs.attention_mask

  return new_batch

chk_name = "chartfc_chartqa_6000_val_loss_1.0426464707541043.pth"
checkpoint = torch.load(f"matcha_checkpoints/{chk_name}")

model.load_state_dict(checkpoint["model_state_dict"])

batch_size = 8
model.to(device);
model.eval();


for split2 in ["seen", "unseen"]:
    res = []
    test_dataset = ChartFCDataset(processor, split="test", split2=split2)
    test_dataloader =  DataLoader(test_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator)

    predictions = []
    for idx, batch in tqdm(enumerate(test_dataloader)):
        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)
        imgnames = batch.pop("imgname")
        header_texts = batch.pop("header_texts")

        generated_ids = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=128)
        predicted_answers = processor.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
        labels = processor.tokenizer.batch_decode(labels,skip_special_tokens=True)

        for img, header, generated_id, answer, label in zip(imgnames,header_texts, generated_ids, predicted_answers, labels):
            predictions.append({"imgname":img, "query":header, "generated_ids":  generated_id.cpu().numpy(), "decoded_answer": answer, "label":label})
    
    with open(f"{chk_name}_{split2}.pkl", "wb") as f:
       f.write(pickle.dumps(predictions)) 

    exact_match_metric = load("exact_match")

    exact_match_metric.compute(predictions=[item["decoded_answer"].split()[0] for item in predictions], references=[item["label"].split()[0] for item in predictions])

    preds = [1 if item["decoded_answer"].split()[0] == "Yes." else 0 for item in predictions]

    labs = [1 if item["label"].split()[0] == "Yes." else 0 for item in predictions]

    s = f1_score(labs,preds, average="macro")
    res.append(("f1-macro", s))

    s = f1_score(labs,preds, average="micro")
    res.append(("f1-micro", s))

    rouge = load('rouge')

    results = rouge.compute(predictions=[" ".join(item["decoded_answer"].split("Explanation: ")[1:]) for item in predictions], references=[" ".join(item["label"].split("Explanation: ")[1:]) for item in predictions])
    res.append(("rouge", results))

    bleurt = load("bleurt", module_type="metric", checkpoint="BLEURT-20")

    results = bleurt.compute(predictions=[" ".join(item["decoded_answer"].split("Explanation: ")[1:]) for item in predictions], references=[" ".join(item["label"].split("Explanation: ")[1:]) for item in predictions])
    res.append(("bleurt", np.array(results["scores"]).mean()))

    bleu = load("bleu")

    results = bleu.compute(predictions=[" ".join(item["decoded_answer"].split("Explanation: ")[1:]) for item in predictions], references=[" ".join(item["label"].split("Explanation: ")[1:]) for item in predictions])
    res.append(("bleu", results))

    meteor = load('meteor')

    results = meteor.compute(predictions=[" ".join(item["decoded_answer"].split("Explanation: ")[1:]) for item in predictions], references=[" ".join(item["label"].split("Explanation: ")[1:]) for item in predictions])
    res.append(("meteor", results))

    bertscore = load("bertscore")

    results = bertscore.compute(predictions=[" ".join(item["decoded_answer"].split("Explanation: ")[1:]) for item in predictions], references=[" ".join(item["label"].split("Explanation: ")[1:]) for item in predictions], lang='en', device=device, model_type="microsoft/deberta-xlarge-mnli")
    res.append(("bertscore", {k:np.array(v).mean() if k != "hashcode" else v for k,v in results.items()}))

    print(split2)
    print(res)
