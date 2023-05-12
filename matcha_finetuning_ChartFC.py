import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup
import json
from PIL import Image
import os
import torch
from evaluate import load
from itertools import cycle

processor = AutoProcessor.from_pretrained("google/matcha-base")
model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-base")
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_PATCHES = 2048


class ChartFCDataset(Dataset):
    def __init__(self, processor, root_dir="ChartFC", split='train', split2="both"):
        """
        Args:
            root_dir (string): Directory with all the ChartQA data.
            split (string): Which split to load ("train" or "val" or "test").
        """
        self.processor = processor
        self.root_dir = root_dir
        self.split = split
        self.image_dir = root_dir

        if split == 'train':
          with open("ChartFC/train/train.json", "r") as f:
            self.data = json.load(f)
        elif split=="val":
          with open("ChartFC/val/val.json", "r") as f:
            self.data = json.load(f)
        elif split == "test":
          if split2 == "both":
            with open("ChartFC/test/test.json", "r") as f:
              seen= json.load(f)
            with open("ChartFC/test/test_unseen.json", "r") as f:
              unseen = json.load(f)
            self.data = seen + unseen
          elif split2 == "seen":
            with open("ChartFC/test/test.json", "r") as f:
              self.data = json.load(f)
          elif split2 == "unseen":
            with open("ChartFC/test/test_unseen.json", "r") as f:
              self.data = json.load(f)

        for example in self.data:
          try:
            imgname = os.path.basename(example["chart_img"])
            Image.open(f"{self.image_dir}/{imgname}").convert('RGB')
          except Exception as e:
            print(example["chart_img"])
            self.data.remove(example)
          example["label"] = f"Yes because {example['explanation']}" if example["label"] == "TRUE" else f"No because {example['explanation']}"
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        qa = self.data[idx]
        # Load image
        imgname = os.path.basename(qa["chart_img"])
        qa["imgname"] = imgname
        qa["image"] = Image.open(f"{self.image_dir}/{imgname}").convert('RGB')
        return qa

def get_query(claim):
  question = "Does the chart support the claim:"
  suffix = "(Yes/No)?"
  return f"{question} {claim} {suffix}"


def collator(batch):
  new_batch = {"flattened_patches":[], "attention_mask":[]}
  images = [item["image"] for item in batch]
  header_texts = [get_query(item['claim']) for item in batch]
  label_texts = [f"{item['label']}" for item in batch] # because {item['explanation']}
  
  inputs = processor(images=images, text=header_texts, return_tensors="pt")
  labels = processor.tokenizer(label_texts, return_tensors="pt", padding=True)
  new_batch["labels"] = labels.input_ids
  new_batch["flattened_patches"] = inputs["flattened_patches"]
  new_batch["attention_mask"] = inputs["attention_mask"]
  new_batch["header_texts"] = header_texts
  new_batch["imgname"] = [item["imgname"] for item in batch]


  return new_batch

batch_size = 4

train_dataset = ChartFCDataset(processor, split='train')
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator)
val_dataset = ChartFCDataset(processor, split='val')
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator)

training_steps = 40000 
checkpoint_steps = 1000

optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=1e-5, weight_decay=1e-07)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=training_steps)
#model = torch.nn.DataParallel(model)
checkpoint_dir = './matcha_checkpoints'

# Check if the directory exists
if not os.path.exists(checkpoint_dir):
    # Create the directory if it doesn't exist
    os.makedirs(checkpoint_dir)

#checkpoint = torch.load(f"{checkpoint_dir}/checkpoint_training_step_6500_val_loss_1.8870208201309044.pth")

#model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.train()

prev_val_loss = float("inf")
prev_checkpoint_name = ""



current_step=0
generator = iter(train_dataloader)
for idx in range(current_step, training_steps):
  try:
    # Samples the batch
    batch = next(generator)
  except StopIteration:
    # restart the generator if the previous generator is exhausted.
    generator = iter(train_dataloader)
    batch = next(generator)
  labels = batch.pop("labels").to(device)
  flattened_patches = batch.pop("flattened_patches").to(device)
  attention_mask = batch.pop("attention_mask").to(device)
  
  masked_labels = labels.masked_fill(labels == model.config.pad_token_id, -100)

  outputs = model(flattened_patches=flattened_patches,
                  attention_mask=attention_mask,
                  labels=masked_labels)
  loss = outputs.loss
  loss.backward()

  print(f"Step {idx+1}/{training_steps} - Loss: {loss.item()}")

  optimizer.step()
  optimizer.zero_grad()
  scheduler.step()

  if (idx+1) % checkpoint_steps == 0:
      with torch.no_grad():
        model.eval()

        val_loss = []
        val_batch_size = []
        for batch in val_dataloader:
          labels = batch.pop("labels").to(device)
          flattened_patches = batch.pop("flattened_patches").to(device)
          attention_mask = batch.pop("attention_mask").to(device)

          masked_labels = labels.masked_fill(labels == model.config.pad_token_id, -100)

          outputs = model(flattened_patches=flattened_patches,
                    attention_mask=attention_mask,
                    labels=masked_labels)
          loss = outputs.loss
          curr_val_loss = loss.item()
          val_loss.append(curr_val_loss)
          val_batch_size.append(labels.size(0))
        
        val_loss_average = sum([v_loss*b_size for v_loss, b_size in zip(val_loss, val_batch_size)]) / sum(val_batch_size)
        print(f"Validation Loss: {val_loss_average}")
        if val_loss_average < prev_val_loss:
          checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
          }
          checkpoint_name = f'{checkpoint_dir}/ChartFC_checkpoint_training_step_{idx+1}_val_loss_{val_loss_average}.pth'
          torch.save(checkpoint, checkpoint_name)
          try:
            os.remove(prev_checkpoint_name)
          except Exception:
            pass
          prev_checkpoint_name = checkpoint_name
          prev_val_loss = val_loss_average

        model.train()
  
  if idx+1 == training_steps:
    break