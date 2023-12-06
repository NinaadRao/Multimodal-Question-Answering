import torch
from torch.utils.data import DataLoader
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
import json
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('--exp_num', type=int, help='Experiment number for logging')
parser.add_argument('--finetuned', type=str, default=False, help='Do you want to inference with finetuned model?')
parser.add_argument('--instruct', type=str, default=False, help='Do you want to inference with instruct model or base model?')
parser.add_argument('--use_SG', type=str, default=False, help='Do you want to inference with SG finetuned model?')
parser.add_argument('--use_GTSG', type=str, default=False, help='Do you want to inference with GT SG finetuned model?')
parser.add_argument('--resume', type=str, default=False, help='Are you resuming previous run?')

# Parse the arguments
args = parser.parse_args()

print('Loaded args')
args.finetuned = True if args.finetuned.lower() in ['1','true'] else False
args.instruct = True if args.instruct.lower() in ['1','true'] else False
args.use_SG = True if args.use_SG.lower() in ['1','true'] else False
args.use_GTSG = True if args.use_GTSG.lower() in ['1','true'] else False
args.resume = True if args.resume.lower() in ['1','true'] else False


with open('AWS/huggingface.json') as f:
  hftokens = json.load(f)
  hftoken_read = hftokens['read']
  hftoken_write = hftokens['write']

quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      llm_int8_skip_modules=["lm_head", "embed_tokens"],
  )

def get_finetuned_model(instruct=True,use_sg=False,use_gtsg=False):
  if instruct:
      model_id = "HuggingFaceM4/idefics-9b-instruct"
      if use_sg:
          if use_gtsg:
            adapter_id = "pragnyas/IDEFICS-9b-instruct-GQA_GTSG-1ep"
          else:
            adapter_id = "pragnyas/IDEFICS-9b-instruct-GQA_GenSG-1ep"
      else:
          adapter_id = "pragnyas/IDEFICS-9b-instruct-GQA_NoSG-1ep"
  else:
      model_id = "HuggingFaceM4/idefics-9b"
      if use_sg:
          if use_gtsg:
              adapter_id = "pragnyas/IDEFICS-9b-GQA_SG-full"
          else:
              adapter_id = "pragnyas/IDEFICS-9b-GQA_GenSG-full"
      else:
          adapter_id = "pragnyas/IDEFICS-9b-GQA_noSG-full"

  print(f'\n\nLoading adapter {adapter_id} for model {model_id}...')
  
  config = PeftConfig.from_pretrained(adapter_id, token=hftoken_read)

  model = IdeficsForVisionText2Text.from_pretrained(
      config.base_model_name_or_path,
      quantization_config=quantization_config,
      device_map="auto",
      token=hftoken_read
  )

  processor = AutoProcessor.from_pretrained(config.base_model_name_or_path, token=hftoken_read)
  model = PeftModel.from_pretrained(model, adapter_id, token=hftoken_read)

  return model, processor

def get_unfinetuned_model(instruct=False):
  if instruct:
    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
  else:
    checkpoint = "HuggingFaceM4/idefics-9b"
  print(f'\n\nLoading {checkpoint} ...')
  processor = AutoProcessor.from_pretrained(checkpoint)

  model = IdeficsForVisionText2Text.from_pretrained(
      checkpoint,
      quantization_config=quantization_config,
      device_map="auto"
  )

  return model, processor



enum=args.exp_num
raw_output = dict()
b = 0
save_path = f'Experiments/Exp{enum}'

if args.resume:
  with open(os.path.join(save_path, 'raw_outputs.json')) as f:
    raw_output = json.load(f)
    b = max([int(x) for x in raw_output.keys()])
  print(b'Starting from {b}')

# testdir = 'GQA/test'
testdir = '../../../data/datasets/GQA/images'
# testdir = '/content/drive/MyDrive/Sem3/MQA/Dataset/GQA/val'

# if not os.path.exist(save_path):
#   os.makedirs(save_path)

if args.finetuned:
   model, processor = get_finetuned_model(instruct=args.instruct, use_sg=args.use_SG,use_gtsg=args.use_GTSG) 
else:
   model, processor = get_unfinetuned_model(instruct=args.instruct) 


with open(f'Experiments/Exp{enum}/prompts_batched.json', 'r') as f:
  prompts = json.load(f)

while b<len(prompts):
  p = prompts[str(b)]

  img_names = [x[1] for x in prompts[str(b)]]
  images = [Image.open(os.path.join(testdir,x)) for x in img_names]


  for i in range(len(p)):
    p[i][1] = images[i]

  inputs = processor(p, return_tensors="pt").to("cuda")
  bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

  generated_ids = model.generate(**inputs, max_new_tokens=25, bad_words_ids=bad_words_ids)
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

  raw_output[b] = []
  for answer in generated_text:
    raw_output[b].append(answer)

  b+= 1

  # Write file every 10 batches
  if b%10==0:
    print(f'Saving after batch {b}/{len(prompts)}')
    with open(os.path.join(save_path, 'raw_outputs.json'), 'w+') as f:
      json.dump(raw_output, f)

  for i in range(len(images)):
    images[i].close()
  del images

  # break

with open(os.path.join(save_path, 'raw_outputs.json'), 'w+') as f:
    json.dump(raw_output, f)