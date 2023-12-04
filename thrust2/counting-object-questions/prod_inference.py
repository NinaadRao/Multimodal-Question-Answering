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
# from prod_genSG import SceneGraphGenerator
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Global variables
qa_model = None
processor = None

hf_username = ""  
hf_repo = "countingqa-finetuned-idefics"
hftoken_write = ""
hftoken_read = ""

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
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["lm_head", "embed_tokens"],
    )
    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(model, 'idefics-9b-instruct-tallyqa/best-checkpoint')
    return model, processor

def make_prompt(question, image_path, ret_img=False):
    # global sg_model

    # sg = sg_model.generate_SG(os.path.join(image_path))

    prompt = ["Instruction: Provide an answer to the question. Use the image and details to answer with ONE WORD OR ONE NUMBER ONLY.\n",
        image_path,
        # f"Scene Details: {sg}",
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
    for b_idx in range(len(batched_prompts)):
        prompts = batched_prompts[b_idx]
        img_names = [x[1] for x in prompts]
        images = [Image.open(os.path.join("",x)) for x in img_names]

        for i in range(len(prompts)):
            prompts[i][1] = images[i]

        inputs = processor(prompts, return_tensors="pt").to("cuda")
        bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        generated_ids = qa_model.generate(**inputs, max_new_tokens=25, bad_words_ids=bad_words_ids)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for answer in generated_text:
            outputs['answers'].append(format_answer(answer))


        # Write file every 10 batches
        if b_idx%10==0:
            print(f'Saving after batch {b_idx}/{len(batched_prompts)}')
            with open('raw_outputs_test.json', 'w+') as f:
                json.dump(outputs, f)

        for i in range(len(images)):
            images[i].close()
        del images

        # break

    print(f'Saving after batch {b_idx}/{len(batched_prompts)}')
    with open('raw_outputs_test.json', 'w+') as f:
        json.dump(outputs, f)

    return outputs['answers']

def inferenceOne(question,image_path):
    global qa_model
    global processor
    # global sg_model

    prompt = make_prompt(question,image_path,ret_img=True)
    inputs = processor(prompt, return_tensors="pt").to("cuda")
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = qa_model.generate(**inputs, max_new_tokens=25, bad_words_ids=bad_words_ids)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return format_answer(generated_text[0])


if __name__=='__main__':
    # Get model
    model_id = "HuggingFaceM4/idefics-9b" #TODO: base or instruct?
    adapter_id = "Pratik2411/countingqa-finetuned-idefics" #TODO: name of saved finetuned model

    qa_model, processor = get_model(model_id,adapter_id)
    print('Initialized QA model')
    # sg_model = SceneGraphGenerator()
    # print('Initialized SG model')

    # Set image dir
    imdir = './router_dataset/test_images/test_images_large' # <path to Yujia's dataset>
    csv_file = './router_dataset/mqa_test_large.csv' # <path to csv file
    batched_prompts = batch_prompts(csv_file,imdir)

    answers = inference(batched_prompts, imdir)