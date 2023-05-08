import numpy as np
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import json
from PIL import Image
import os
import torch
import cv2

def sharpen_image(img):
    img = np.array(img)

    # Define the kernel size for blurring the image
    kernel_size = (3, 3)

    # Define the amount of sharpening to be applied
    sharpen_strength = 3

    # Create a Gaussian filter kernel for blurring the image
    kernel = cv2.getGaussianKernel(kernel_size[0], 0)
    kernel = np.outer(kernel, kernel.transpose())

    # Subtract the blurred image from the original image to obtain the sharpened image
    blurred_img = cv2.filter2D(img, -1, kernel)
    sharpened_img = cv2.addWeighted(img, 1 + sharpen_strength, blurred_img, -sharpen_strength, 0)
    sharpened_pil_img = Image.fromarray(sharpened_img)
    return sharpened_pil_img

model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
processor = AutoProcessor.from_pretrained("google/deplot")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor.image_processor.is_vqa=False

model.to(device);

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
    except Exception:
        pass

data=np.array(new_data)

np.random.seed(42)

# Shuffle the indices of the data
indices = np.random.permutation(len(data))

# Calculate the number of samples in the training, validation, and testing sets
num_train = int(0.8 * len(data))
num_val = int(0.1 * len(data))

# Split the indices into training, validation, and testing sets
train_indices = indices[:num_train]
val_indices = indices[num_train:num_train+num_val]
test_indices = indices[num_train+num_val:]

train_data = data[train_indices]
val_data = data[val_indices]
test_data = data[test_indices]

len(train_data)

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

for item in train_data:
    filename = os.path.splitext(os.path.basename(item["chart_img"]))[0]
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

category = {"bar_horizontal":[], "bar_vertical":[],"line_chart":[],"pie_chart":[],"mixed":[]}

for item in train_data:
    category[item["chart_type"]].append(item)

# for item in np.random.choice(category["bar_horizontal"], 10, replace=False):
#     imgname = os.path.basename(item["chart_img"])
#     img = Image.open(f"ChartFC/{imgname}").convert('RGB')
#     sharpened_img = sharpen_image(img)
#     normal_inputs = processor(images=img, return_tensors="pt")
#     sharpened_input = processor(images=sharpened_img, return_tensors="pt")
#     normal_generated_ids = model.generate(flattened_patches=normal_inputs["flattened_patches"].to(device), attention_mask=normal_inputs["attention_mask"].to(device), max_new_tokens=512)
#     normal_predicted_answer = processor.tokenizer.batch_decode(normal_generated_ids,skip_special_tokens=True)[0].replace("<0x0A>", "\n")
#     sharpened_generated_ids = model.generate(flattened_patches=sharpened_input["flattened_patches"].to(device), attention_mask=sharpened_input["attention_mask"].to(device), max_new_tokens=512)
#     sharpened_predicted_answer = processor.tokenizer.batch_decode(sharpened_generated_ids,skip_special_tokens=True)[0].replace("<0x0A>", "\n")
#     img.show()
#     print(normal_predicted_answer)
#     sharpened_img.show()
#     print(sharpened_predicted_answer)
#
# for item in np.random.choice(category["bar_vertical"], 10, replace=False):
#   imgname = os.path.basename(item["chart_img"])
#   img = Image.open(f"ChartFC/{imgname}").convert('RGB')
#   sharpened_img = sharpen_image(img)
#   normal_inputs = processor(images=img, return_tensors="pt")
#   sharpened_input = processor(images=sharpened_img, return_tensors="pt")
#   normal_generated_ids = model.generate(flattened_patches=normal_inputs["flattened_patches"].to(device), attention_mask=normal_inputs["attention_mask"].to(device), max_new_tokens=512)
#   normal_predicted_answer = processor.tokenizer.batch_decode(normal_generated_ids,skip_special_tokens=True)[0].replace("<0x0A>", "\n")
#   sharpened_generated_ids = model.generate(flattened_patches=sharpened_input["flattened_patches"].to(device), attention_mask=sharpened_input["attention_mask"].to(device), max_new_tokens=512)
#   sharpened_predicted_answer = processor.tokenizer.batch_decode(sharpened_generated_ids,skip_special_tokens=True)[0].replace("<0x0A>", "\n")
#   img.show()
#   print(normal_predicted_answer)
#   sharpened_img.show()
#   print(sharpened_predicted_answer)
#
# for item in np.random.choice(category["line_chart"], 10, replace=False):
#   imgname = os.path.basename(item["chart_img"])
#   img = Image.open(f"ChartFC/{imgname}").convert('RGB')
#   sharpened_img = sharpen_image(img)
#   normal_inputs = processor(images=img, return_tensors="pt")
#   sharpened_input = processor(images=sharpened_img, return_tensors="pt")
#   normal_generated_ids = model.generate(flattened_patches=normal_inputs["flattened_patches"].to(device), attention_mask=normal_inputs["attention_mask"].to(device), max_new_tokens=512)
#   normal_predicted_answer = processor.tokenizer.batch_decode(normal_generated_ids,skip_special_tokens=True)[0].replace("<0x0A>", "\n")
#   sharpened_generated_ids = model.generate(flattened_patches=sharpened_input["flattened_patches"].to(device), attention_mask=sharpened_input["attention_mask"].to(device), max_new_tokens=512)
#   sharpened_predicted_answer = processor.tokenizer.batch_decode(sharpened_generated_ids,skip_special_tokens=True)[0].replace("<0x0A>", "\n")
#   img.show()
#   print(normal_predicted_answer)
#   sharpened_img.show()
#   print(sharpened_predicted_answer)
#
# for item in np.random.choice(category["pie_chart"], 10, replace=False):
#   imgname = os.path.basename(item["chart_img"])
#   img = Image.open(f"ChartFC/{imgname}").convert('RGB')
#   sharpened_img = sharpen_image(img)
#   normal_inputs = processor(images=img, return_tensors="pt")
#   sharpened_input = processor(images=sharpened_img, return_tensors="pt")
#   normal_generated_ids = model.generate(flattened_patches=normal_inputs["flattened_patches"].to(device), attention_mask=normal_inputs["attention_mask"].to(device), max_new_tokens=512)
#   normal_predicted_answer = processor.tokenizer.batch_decode(normal_generated_ids,skip_special_tokens=True)[0].replace("<0x0A>", "\n")
#   sharpened_generated_ids = model.generate(flattened_patches=sharpened_input["flattened_patches"].to(device), attention_mask=sharpened_input["attention_mask"].to(device), max_new_tokens=512)
#   sharpened_predicted_answer = processor.tokenizer.batch_decode(sharpened_generated_ids,skip_special_tokens=True)[0].replace("<0x0A>", "\n")
#   img.show()
#   print(normal_predicted_answer)
#   sharpened_img.show()
#   print(sharpened_predicted_answer)
#
# for item in np.random.choice(category["mixed"], len(category["mixed"]), replace=False):
#   imgname = os.path.basename(item["chart_img"])
#   img = Image.open(f"ChartFC/{imgname}").convert('RGB')
#   sharpened_img = sharpen_image(img)
#   normal_inputs = processor(images=img, return_tensors="pt")
#   sharpened_input = processor(images=sharpened_img, return_tensors="pt")
#   normal_generated_ids = model.generate(flattened_patches=normal_inputs["flattened_patches"].to(device), attention_mask=normal_inputs["attention_mask"].to(device), max_new_tokens=512)
#   normal_predicted_answer = processor.tokenizer.batch_decode(normal_generated_ids,skip_special_tokens=True)[0].replace("<0x0A>", "\n")
#   sharpened_generated_ids = model.generate(flattened_patches=sharpened_input["flattened_patches"].to(device), attention_mask=sharpened_input["attention_mask"].to(device), max_new_tokens=512)
#   sharpened_predicted_answer = processor.tokenizer.batch_decode(sharpened_generated_ids,skip_special_tokens=True)[0].replace("<0x0A>", "\n")
#   img.show()
#   print(normal_predicted_answer)
#   sharpened_img.show()
#   print(sharpened_predicted_answer)
#
