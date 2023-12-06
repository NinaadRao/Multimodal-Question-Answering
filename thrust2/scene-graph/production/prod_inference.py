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
import csv
from prod_genSG import SceneGraphGenerator
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Global variables
qa_model = None
sg_model = None
processor = None

# Read access tokens for Huggingface
with open('../AWS/huggingface.json') as f:
  hftokens = json.load(f)
  hftoken_read = hftokens['read']
  hftoken_write = hftokens['write']


# Set quantization config
quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      llm_int8_skip_modules=["lm_head", "embed_tokens"],
  )



# Load model
def get_model(model_id,adapter_id):
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

def make_prompt(question, image_path, ret_img=False):
    global sg_model

    sg = sg_model.generate_SG(os.path.join(image_path))

    prompt = ["Instruction: Provide an answer to the question. Use the image and details to answer with ONE WORD ONLY.\n",
        image_path,
        f"Scene Details: {sg}",
        f"Question: {question} Answer:"
    ]

    if ret_img:
        im = Image.open(image_path)
        prompt[1] = im
    
    return prompt

# Get scene graph and make prompts
def batch_prompts(csv_file, imdir, batchsize=4):
    b_idx = 0
    batched_prompts = {0:[]}

    with open(csv_file, 'r') as f:
        df = csv.DictReader(f)

        for row in df:
            if len(batched_prompts[b_idx])==batchsize:
                b_idx += 1
                batched_prompts[b_idx] = []

            prompt = make_prompt(row['question'], image_path=os.path.join(imdir,row['image_path']))
            batched_prompts[b_idx].append(prompt)
            if b_idx%100==0:
                print(f'Finished batching batch {b_idx}')
    return batched_prompts

def format_answer(answer):
    ans = re.findall(r'Answer: (.*)', answer)
    if not ans:
        return ''
    ans = ans[0]

    # Clean answer:
    ans = ans.lower()

    # Remove punctuation:
    ans = re.sub(r'[^\w\s]','',ans)

    # Remove stopwords using nltk:
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.word_tokenize(ans)
    ans = [w for w in word_tokens if not w in stop_words]

    # Make singular
    lemmatizer = nltk.stem.WordNetLemmatizer()
    ans = [lemmatizer.lemmatize(word, pos='n') for word in ans]

    ans = ' '.join(ans)
    return ans

# Get inference for whole dataset
def inference(batched_prompts, imdir):
    global qa_model
    global processor

    outputs = {'answers':[]}
    # with open('raw_outputs_pragnyas.json','r') as f:
    #     outputs = json.load(f)
    
    key = list(batched_prompts.keys())
    # print('\n\n\n',min(key),max(key),'\n\n\n')
    for b_idx in range(len(batched_prompts)):
        prompts = batched_prompts[b_idx]
        img_names = [x[1] for x in prompts]
        images = [Image.open(x) for x in img_names]

        for i in range(len(prompts)):
            prompts[i][1] = images[i]

        inputs = processor(prompts, return_tensors="pt").to("cuda")
        bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        generated_ids = qa_model.generate(**inputs, max_new_tokens=25, bad_words_ids=bad_words_ids)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for answer in generated_text:
            outputs['answers'].append(answer)


        # Write file every 10 batches
        if b_idx%10==0:
            print(f'Saving after batch {b_idx}/{len(batched_prompts)}')
            with open('raw_outputs_pragnyas_final_finetuned_train.json', 'w+') as f:
                json.dump(outputs, f)

        for i in range(len(images)):
            images[i].close()
        del images

    print(f'Saving after batch {b_idx}/{len(batched_prompts)}')
    with open('raw_outputs_pragnyas_final_finetuned_train.json', 'w+') as f:
        json.dump(outputs, f)

    return outputs['answers']

def inferenceOne(question,image_path):
    global qa_model
    global processor
    global sg_model

    prompt = make_prompt(question,image_path,ret_img=True)
    inputs = processor(prompt, return_tensors="pt").to("cuda")
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = qa_model.generate(**inputs, max_new_tokens=25, bad_words_ids=bad_words_ids)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # print(generated_text[0])
    return format_answer(generated_text[0])


if __name__=='__main__':
    # Get model
    model_id = "HuggingFaceM4/idefics-9b-instruct" #TODO: base or instruct?
    # adapter_id = "Pratik2411/countingqa-finetuned-idefics" #TODO: name of saved finetuned model
    # adapter_id = "pragnyas/IDEFICS-9b-instruct-GQA_GenSG-full"
    adapter_id = "pragnyas/IDEFICS-9b-instruct-GQA_GenSG-1ep"
    
    qa_model, processor = get_model(model_id,adapter_id)
    print('Initialized QA model')
    sg_model = SceneGraphGenerator()
    print('Initialized SG model')

    # Set image dir
    imdir = './data/train_images' # <path to Yujia's dataset>
    csv_file = './data/router_train.csv' # <path to csv file>

    # Sanity test make_prompt function
    # prompt = make_prompt('Are these indian or african elephants?', os.path.join(imdir,'COCO_train2014_000000000061.jpg'))
    # print(prompt)
    # answer = inferenceOne('Are these indian or african elephants?',os.path.join(imdir,'COCO_train2014_000000000061.jpg'))
    # print('\n\n', answer)
    
    batched_prompts = batch_prompts(csv_file,imdir,2)
    with open('batched_prompt_pragnyas_final_finetuned_train.json','w+') as f:
        json.dump(batched_prompts,f)
    # with open('batched_prompt_pragnyas.json','r') as f:
        # batched_prompts = json.load(f)
    answers = inference(batched_prompts, imdir)