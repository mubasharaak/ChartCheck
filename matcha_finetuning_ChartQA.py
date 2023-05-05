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

class ChartQADataset(Dataset):
    def __init__(self, processor, root_dir="ChartQA Dataset", split='train', split2="both"):
        """
        Args:
            root_dir (string): Directory with all the ChartQA data.
            split (string): Which split to load ("train" or "val" or "test").
            split2 (string): Which split to load ("both" or "augmented" or "human") within the first split.
        """
        self.processor = processor
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(self.root_dir, self.split, 'png')
        
        self.qa_augmented = []
        self.qa_human = []
        # Load questions and answers
        with open(os.path.join(self.root_dir, self.split, f'{self.split}_augmented.json'), 'r',  encoding='utf-8') as f:
            self.qa_augmented = json.load(f)
        with open(os.path.join(self.root_dir, self.split, f'{self.split}_human.json'), 'r', encoding='utf-8') as f:
            self.qa_human = json.load(f)

        if split2 == "both":
            self.data = self.qa_augmented + self.qa_human
        elif split2 == "augmented":
            self.data = self.qa_augmented
        elif split2 == "human":
            self.data = self.qa_human
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        qa = self.data[idx]
        # Load image
        qa["image"] = Image.open(f"{self.image_dir}/{qa['imgname']}").convert('RGB')
        return qa

def collator(batch):
  new_batch = {"flattened_patches":[], "attention_mask":[]}
  images = [item["image"] for item in batch]
  header_texts = [item["query"] for item in batch]
  label_texts = [item['label'] for item in batch]
  
  inputs = processor(images=images, text=header_texts, return_tensors="pt")
  labels = processor.tokenizer(label_texts, return_tensors="pt", padding=True)
  new_batch["labels"] = labels.input_ids
  new_batch["flattened_patches"] = inputs["flattened_patches"]
  new_batch["attention_mask"] = inputs["attention_mask"]


  return new_batch

batch_size = 4

train_dataset = ChartQADataset(processor, split='train')
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator)
val_dataset = ChartQADataset(processor, split='val', split2="human")
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator)

training_steps = 40000 
checkpoint_steps = 1000

optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=1e-5, weight_decay=1e-07)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=training_steps)
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
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
          checkpoint_name = f'{checkpoint_dir}/checkpoint_training_step_{idx+1}_val_loss_{val_loss_average}.pth'
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