import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
import torchvision.transforms as transforms
import wandb
import os
import argparse
parser = argparse.ArgumentParser(
                    prog='train-idefics',
                    description='script to train idefics-9b')

parser.add_argument('-w', '--wandb', default= 'idefics-9b') 
parser.add_argument('-r', '--ratio', type=float, default = 0.5)
parser.add_argument('-d', '--dir', default = 'idefics-latest')
parser.add_argument('-m', '--model', default = 'idefics-90')
args = parser.parse_args()

os.environ["WANDB_PROJECT"]="MQA"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
wandb.login(key = "6303eb738fdf6199f76497134963d53a7f8cd9be")
device = "cuda" if torch.cuda.is_available() else "cpu"



# checkpoint = "HuggingFaceM4/tiny-random-idefics"
checkpoint = "HuggingFaceM4/idefics-9b"

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
        prompts.append(
            [
                Image.open(example_batch['image_path'][i]),
                "Question: {} Answer: {}</s>".format(example_batch['question'][i], example_batch['answer'][i]),
            ],
        )
    inputs = processor(prompts, transform=image_transform, return_tensors="pt").to(device)
    inputs["labels"] = inputs["input_ids"]
    return inputs


# load and prepare dataset
ds1 = load_dataset('json', data_files = '/data/user_data/naveensu/CLEVR_v1_processed.json')
ds2 = load_dataset('json', data_files = '/data/user_data/naveensu/vqa/vqacp_v2_train_processed.json')
vqa_size = len(ds2['train'])
clevr_size = len(ds1['train'])
training_size = 699989
ds1 = ds1['train'].train_test_split(test_size = clevr_size - int(args.ratio * training_size), seed = 69)
ds2 = ds2['train'].train_test_split(test_size = vqa_size - int((1 - args.ratio) * training_size), seed = 69)
train_ds = concatenate_datasets([ds1['train'], ds2['train']])
eval_ds = Dataset.from_dict(concatenate_datasets([ds1['test'], ds2['test']]).shuffle(seed = 69)[0:1000])
train_ds.set_transform(ds_transforms)
eval_ds.set_transform(ds_transforms)


model_name = checkpoint.split("/")[1]
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, config)


training_args = TrainingArguments(
    output_dir=args.dir,
    learning_rate=2e-4,
    fp16=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=100,
    dataloader_pin_memory=False,
    save_total_limit=45,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1000,
    eval_steps=100,
    logging_steps=15,
    num_train_epochs=1,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
    load_best_model_at_end=True,
    optim="paged_adamw_8bit",
    report_to="wandb",
    run_name=args.wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)
# trainer.train()
trainer.train(resume_from_checkpoint = '/data/user_data/naveensu/' + args.model)
