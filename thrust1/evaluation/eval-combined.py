import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
import torchvision.transforms as transforms
import wandb
import os
from peft import PeftModel
import json
import pandas as pd
import time
import logging
import argparse
parser = argparse.ArgumentParser(
                    prog='eval-idefics',
                    description='script to eval idefics-9b')

parser.add_argument('-m', '--model', default= 'idefics-50') 
parser.add_argument('-o', '--output', default = 'latest-eval.csv')
parser.add_argument('-i', '--image', default = 'train_images_small')
parser.add_argument('-d', '--data', default = 'router_train_small.csv')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


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
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="auto")

model = PeftModel.from_pretrained(model, args.model)


def generate_output_answer(prompts):
    
    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=128)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    return generated_text

def evaluate_result(predicted_answer, ground_truth_answer):
    dict_map = {'one' : '1',
                'two': '2',
                'three': '3',
                'four': '4',
                'five': '5',
                'six': '6',
                'seven': '7',
                'eight': '8',
                'nine': '9'
               }
    ground_truth_answer = str(ground_truth_answer)
    predicted_answer = predicted_answer.lower()
    splitted = predicted_answer.split()

    for i in range(len(splitted)):
        if splitted[i] in dict_map:
            splitted[i] = dict_map[splitted[i]]
    predicted_answer = ' '.join(splitted)
    if ground_truth_answer.lower() in predicted_answer.lower():
        return True
    else:
        return False

total_questions = 0
total_correct = 0

output_dict = {'question': [], 
               'image_filename': [],
               'ground_truth_ans': [], 
               'predicted_ans': [], 
               'correct': [],
              }

base_url = '/data/user_data/naveensu/' + args.image
df = pd.read_csv('/data/user_data/naveensu/' + args.data)
prompts = []
ground_truth_answers = []
predicted_answers = []
for i, item in df.iterrows():
    start = time.time()
    total_questions += 1

    question = item['question']
    img_filename = item['image_path']
    image_object_url = base_url + '/' + img_filename
    ground_truth_answer = item['answer']
    ground_truth_answers.append(ground_truth_answer)

    img = Image.open(image_object_url)

    prompts.append([img, "Question: " + question  +  "Answer: "])

   
    output_dict['question'].append(question)
    output_dict['image_filename'].append(img_filename)
    output_dict['ground_truth_ans'].append(ground_truth_answer)

    if total_questions % 128 == 0:

        predicted_answer = generate_output_answer(prompts)

        for ans, ground_truth in zip(predicted_answer, ground_truth_answers):
            # print(ans)
            ans = ans.split("Answer: ")[1]
            output_dict['predicted_ans'].append(ans)

            if evaluate_result(ans, ground_truth):
                total_correct += 1
                output_dict['correct'].append(1)
            else:
                output_dict['correct'].append(0)

        print(f"Completed {total_questions}")

        prompts = []
        ground_truth_answers = []
        pd.DataFrame(output_dict).to_csv(args.output, index = False)

    print(time.time() - start)

for ans, ground_truth in zip(predicted_answer, ground_truth_answers):
    # print(ans)
    ans = ans.split("Answer: ")[1]
    output_dict['predicted_ans'].append(ans)
    # print(ans)

    if evaluate_result(str(ans), str(ground_truth)):
        total_correct += 1
        output_dict['correct'].append(1)
    else:
        output_dict['correct'].append(0)

print(f"Completed {total_questions}")

df = pd.DataFrame(output_dict)
df.to_csv(args.output, index = False)
