import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
import torchvision.transforms as transforms
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
DATADIR = '../../../data/datasets/GQA/images'
# checkpoint = "HuggingFaceM4/tiny-random-idefics"
checkpoint = "HuggingFaceM4/idefics-9b-instruct"
# checkpoint = "HuggingFaceM4/idefics-9b"

# Here we skip some special modules that can't be quantized properly
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "embed_tokens"],
)

processor = AutoProcessor.from_pretrained(checkpoint)
# Simply take-off the quantization_config arg if you want to load the original model
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="auto")

# with open('Dataset/GQA/sceneGraphs/all_relbal_sg_tuples_lt100.json') as f:
#     sg_data = json.load(f)

with open('Dataset/GQA/gen_sceneGraphs/train_relbal_sg_tuples.json') as f:
    sg_data = json.load(f)

with open('Dataset/GQA/gen_sceneGraphs/val_relbal_sg_tuples.json') as f:
    sg_data.update(json.load(f))

def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

def ds_transforms(example_batch):
    image_size = processor.image_processor.image_size
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    image_transform = transforms.Compose([
        convert_to_rgb,
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    prompts = []
    for i in range(len(example_batch['question'])):
        # We split the captions to avoid having very long examples, which would require more GPU ram during training
        im = Image.open(os.path.join(DATADIR, example_batch['image'][i]))
        prompts.append(
            [
                "Instruction: Provide an answer to the question. Use the image to answer with ONE WORD ONLY.\n",
                im.copy(),
                f"Question: {example_batch['question'][i]} Answer:{example_batch['answer'][i]}"
            ],
        )
        im.close()

    inputs = processor(prompts, transform=image_transform, return_tensors="pt").to(device)

    inputs["labels"] = inputs["input_ids"]
    # for attribute in inputs:
    #     print(attribute, inputs[attribute].device)

    return inputs

def ds_transforms_with_sg(example_batch):
    image_size = processor.image_processor.image_size
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    image_transform = transforms.Compose([
        convert_to_rgb,
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    prompts = []
    for i in range(len(example_batch['question'])):
        # We split the captions to avoid having very long examples, which would require more GPU ram during training
        im = Image.open(os.path.join(DATADIR, example_batch['image'][i]))
        prompts.append(
            [
                "Instruction: Provide an answer to the question. Use the image and scene details to answer with ONE WORD ONLY.\n",
                im.copy(),
                f"Scene details:{sg_data[example_batch['image'][i]]}\n",
                f"Question: {example_batch['question'][i]} Answer:{example_batch['answer'][i]}"
            ],
        )
        im.close()

    inputs = processor(prompts, transform=image_transform, return_tensors="pt").to(device)

    inputs["labels"] = inputs["input_ids"]
    # for attribute in inputs:
    #     print(attribute, inputs[attribute].device)

    return inputs

  
with open('AWS/huggingface.json') as f:
  hftokens = json.load(f)
  hftoken_read = hftokens['read']
  hftoken_write = hftokens['write']

# load and prepare dataset
# ds = load_dataset("TheFusion21/PokemonCards")
# ds = load_dataset("pragnyas/Balanced_Relational_GQA",token=hftoken_read)
# ds = load_dataset("pragnyas/Balanced_Relational_GQA_full", token=hftoken_read)
ds = load_dataset("pragnyas/Balanced_Relational_GQA_SGFinetune", token=hftoken_read)
train_ds = ds["train"]
eval_ds = ds["validation"]
# train_ds.set_transform(ds_transforms)
# eval_ds.set_transform(ds_transforms)
train_ds.set_transform(ds_transforms_with_sg)
eval_ds.set_transform(ds_transforms_with_sg)

# for key in train_ds.features:
#     tensor = train_ds[key]
#     # train_ds[key] = tensor.to('cuda:0')
#     # print('Moved {}')
#     print(key, tensor.device)

# for key in eval_ds.features:
#     tensor = eval_ds[key]
#     print(key, tensor.device)
    # eval_ds[key] = tensor.to('cuda:0')

# train_ds.to('cuda:0')
# eval_ds.to('cuda:0')

print('\n\n\n')
print(train_ds)
print('\n\n\n')

model_name = checkpoint.split("/")[1]
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, config)
# model = model.to('cuda:0')

print('\n\n\n')
model.print_trainable_parameters()
print(model.device)
print('\n\n\n')

training_args = TrainingArguments(
    output_dir=f"IDEFICS-9b-instruct-GQA_GenSG-1ep",
    learning_rate=2e-4,
    fp16=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=100,
    dataloader_pin_memory=False,
    save_total_limit=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # save_steps=1,
    # eval_steps=1,
    # logging_steps=1,
    # max_steps=10,
    num_train_epochs = 1,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
    load_best_model_at_end=True,
    report_to=None,
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()

model.push_to_hub("pragnyas/IDEFICS-9b-instruct-GQA_GenSG-1ep",private=False, token=hftoken_write)