import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import accelerate
import json
from tqdm.autonotebook import tqdm
import numpy as np
from peft import LoraConfig, get_peft_model
import pandas as pd


def load_model(checkpoint):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
    )
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=quantization_config, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)
    return model, processor


# return a list of answers to a question
def process_answers(answer_list):
    res_answer = []
    for p in answer_list:
        res_answer.append(p['answer'])
    return res_answer


def process_question_file(question_path):
    #format: question_id: (image_id, question)
    question_dict = {}
    question_ids = []
    # process question file
    # Open the JSON file in read mode
    with open(question_path, 'r') as json_file:
        # Read the JSON data into a Python dictionary or list
        question_file = json.load(json_file)
    questions = question_file['questions']
    for p in questions:
        image_id = p['image_id']
        question_id = p['question_id']
        question = p['question']
        question_dict[question_id] = (image_id, question)
        question_ids.append(question_id)
    return question_dict, question_ids


# convert image_id to corresponding url in gcp
# incorrect right now, to be fixed in the future
# val mode
def image_to_url(question_dict):
    # example url: https://storage.googleapis.com/coco_val/val2014/COCO_val2014_000000000042.jpg
    # example url S3: https://okvqa11632.s3.amazonaws.com/val2014/COCO_val2014_000000000042.jpg
    image_url_dict = {}
    for question_id, (image_id, question) in question_dict.items():
        padded_image_id = f"{image_id:012d}"
        # image_url = "https://okvqa11632.s3.amazonaws.com/val2014/COCO_val2014_"+padded_image_id+".jpg"
        image_url = "https://okvqa11632.s3.amazonaws.com/train2014/COCO_train2014_"+padded_image_id+".jpg"
        image_url_dict[image_id] = image_url
    return image_url_dict


def process_annotation_file(annotation_path, question_dict):
    # format: question_id: (image_id, question, answer_list)
    annotation_dict = {}
    # final output: image, question, answer
    # process annotation file
    # Open the JSON file in read mode
    with open(annotation_path, 'r') as json_file:
        # Read the JSON data into a Python dictionary or list
        anno_file = json.load(json_file)

    annotations = anno_file['annotations']
    for e in annotations:
        image_id = e['image_id']
        question_id = e['question_id']
        question = question_dict[question_id][1]
        answers = e['answers']
        answer_list = process_answers(answers)
        annotation_dict[question_id] = (image_id, question, answer_list)
    return annotation_dict


def construct_prompt(annotation_dict, image_url_dict):
    """
    format:
    prompts = [
    "Instruction: provide an answer to the question. Use the image to answer.\n",
    "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg",
    "Question: What's on the picture? Answer: \n"
    ]
    """

    prompt_list = []
    question_id_list = []

    prompt1 = "Instruction: provide a short answer to the question. Use the image to answer.\n"

    for question_id, (image_id, question, answer_list) in annotation_dict.items():
        prompt2 = image_url_dict[image_id]
        prompt3 = "Question: {} Answer: ".format(question)
        cur_prompt = [prompt1, prompt2, prompt3]
        # cur_prompt = [prompt2, prompt3]
        prompt_list.append(cur_prompt)
        question_id_list.append(question_id)
    return prompt_list, question_id_list



def get_embed(prompt_list, question_id_list, processor, model, question_dict, annotation_dict, save_path):

    data = pd.DataFrame(columns=["question_id"] + [i for i in range(4096)])

    for i in tqdm(range(len(prompt_list))):
        prompts = prompt_list[i]
        question_id = question_id_list[i]
        inputs = processor(prompts, return_tensors="pt").to(device)

        output = model.forward(**inputs, output_hidden_states=True)
        embed = output["hidden_states"][-1][0][-1]

        data.loc[len(data.index)] = [question_id] + embed.tolist()

        # if i == 10:
        #     break
        if (i+1) % 500 == 0:
            data.to_csv(save_path, index=False)

    data.to_csv(save_path, index=False)


def get_embed_local(df, processor, model, image_folder, save_path):

    data = pd.DataFrame(columns=["sample_index"] + [i for i in range(4096)])
    prompt1 = "Instruction: provide a short answer to the question. Use the image to answer.\n"

    for i in tqdm(range(df.shape[0])):
        
        prompt2 = Image.open(image_folder + df.loc[i,"image_path"])
        prompt3 = "Question: {} Answer: ".format(df.loc[i,"question"])
        prompts = [prompt1, prompt2, prompt3]

        inputs = processor(prompts, return_tensors="pt").to(device)

        output = model.forward(**inputs, output_hidden_states=True)
        embed = output["hidden_states"][-1][0][-1]

        data.loc[len(data.index)] = [i] + embed.tolist()

        # if i == 10:
        #     break
        if (i+1) % 500 == 0:
            data.to_csv(save_path, index=False)

    data.to_csv(save_path, index=False)


if __name__ == "__main__":
    
    print("--------------   import done   ---------------")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    #checkpoint = "HuggingFaceM4/tiny-random-idefics"

    model, processor = load_model(checkpoint)
    model.load_adapter("yujiaw2/idefics-9b-instruct-okvqa-v4")

    print("--------------   model loaded   ---------------")

    # question_path =  'data/OpenEnded_mscoco_train2014_questions.json'
    # annotation_path =  'data/mscoco_train2014_annotations.json'
    # question_dict, _ = process_question_file(question_path)
    # annotation_dict = process_annotation_file(annotation_path, question_dict)
    # image_url_dict = image_to_url(question_dict)

    # prompt_list, question_id_list = construct_prompt(annotation_dict, image_url_dict) 
    # print("--------------   question & info loaded   ---------------")

    # get_embed(prompt_list, question_id_list, processor, model, question_dict, annotation_dict, save_path="./data/training_embed.csv")
    df = pd.read_csv("./dataset/router_train_small.csv")
    get_embed_local(df, processor, model, image_folder="", save_path="./data/training_embed_mixed.csv")

    import pdb; pdb.set_trace()
    