# this is a demo of inference of IDEFICS-9B using 4bit-quantization which needs about 7GB of GPU memory
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import accelerate
import json
from tqdm.autonotebook import tqdm
import numpy as np
from collections import Counter
import pandas as pd
import random
from collections import Counter
import sys
import argparse



def load_model(checkpoint):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
    )
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=quantization_config, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)
    return model, processor



def process_caption(caption_path):
  # Open the JSON file for reading
  with open(caption_path, 'r') as json_file:
    # Load the JSON data into a Python dictionary
    data = json.load(json_file)
    # iid_to_capt = json.load(caption_path)
    return data

# return a list of answers to a question
def process_answers(answer_list):
    res_answer = []
    for p in answer_list:
        res_answer.append(p['answer'])
    return res_answer


def process_question_file(question_path):
    #format: question_id: (image_id, question)
    question_dict = {}
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
    return question_dict


# convert image_id to corresponding url in gcp
# incorrect right now, to be fixed in the future
# val mode
def image_to_url(question_dict):
    # example url: https://storage.googleapis.com/coco_val/val2014/COCO_val2014_000000000042.jpg
    # example url S3: https://okvqa11632.s3.amazonaws.com/val2014/COCO_val2014_000000000042.jpg
    image_url_dict = {}
    for question_id, (image_id, question) in question_dict.items():
        padded_image_id = f"{image_id:012d}"
        image_url = "https://okvqa11632.s3.amazonaws.com/val2014/COCO_val2014_"+padded_image_id+".jpg"
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
# following two score is rough, and only for print accuracies during inferring.
def ok_score(answer_list):
    # gt_answers = [a['answer'] for a in gt_answers]
    ans2cnt = Counter(answer_list)
    # sort
    ans2cnt = sorted(ans2cnt.items(), key=lambda x: x[1], reverse=True)
    ans2score = {}
    for ans, cnt in ans2cnt:
        # ans2score[ans] = min(1.0, cnt / 3.0)
        if cnt == 1:
            ans2score[ans] = 0.3
        elif cnt == 2:
            ans2score[ans] = 0.6
        elif cnt == 3:
            ans2score[ans] = 0.9
        else:
            ans2score[ans] = 1.0
    return ans2score

def find_most_ans(answer_list):
    ans2score = ok_score(answer_list)
    most_answer = list(ans2score.keys())[0]
    if most_answer == '':
        most_answer = list(ans2score.keys())[1]
    return most_answer



def get_k_examples_PICa(annotation_dict, iid_to_capt,k):
    example_res = ""
    # Choose k random key-value pairs
    random_items = random.sample(list(annotation_dict.items()), k)
    for question_id, (image_id, question, answer_list) in random_items:
        context = iid_to_capt[str(image_id)]
        example_res += "Context: {}\n Question: {}\n Answer: {}\n".format(context,question,find_most_ans(answer_list))
    return example_res

def get_k_examples_answer_aware(train_annotation_dict,annotation_dict, qid, qid_to_answer_aware_example,k):
    example_res = ""
    # Choose k random key-value pairs
    # format: question_id: (image_id, question, answer_list)
    # random_items = random.sample(list(annotation_dict.items()), k)
    similar_qids = get_similar_qids(qid, qid_to_answer_aware_example, k)
    similar_items = {}
    for similar_qid in similar_qids:
        similar_items[similar_qid] = train_annotation_dict[int(similar_qid)]
        
    for question_id, (image_id, question, answer_list) in similar_items.items():
        context = iid_to_capt[str(image_id)]
        example_res += "Context: {}\n Question: {}\n Answer: {}\n".format(context,question,find_most_ans(answer_list))
    return example_res

def get_k_examples_question_aware(train_annotation_dict,annotation_dict, qid, qid_to_question_aware_example,k):
    example_res = ""
    # Choose k random key-value pairs
    # format: question_id: (image_id, question, answer_list)
    # random_items = random.sample(list(annotation_dict.items()), k)
    similar_qids = get_similar_qids(qid, qid_to_question_aware_example, k)
    similar_items = {}
    for similar_qid in similar_qids:
        similar_items[similar_qid] = train_annotation_dict[int(similar_qid)]
        
    for question_id, (image_id, question, answer_list) in similar_items.items():
        context = iid_to_capt[str(image_id)]
        example_res += "Context: {}\n Question: {}\n Answer: {}\n".format(context,question,find_most_ans(answer_list))
    return example_res


def construct_candidates(k_candidates):
    res = []
    for e in k_candidates:
        answer = e['answer']
        confidence = e['confidence']
        construct = "{} ({})".format(answer, confidence)
        res.append(construct)
    return ",".join(res)

def get_k_examples_candidates(train_annotation_dict,annotation_dict, iid_to_capt, qid_to_topkcands, k, k_cands):
    example_res = ""
    # Choose k random key-value pairs
    random_items = random.sample(list(train_annotation_dict.items()), k)
    for question_id, (image_id, question, answer_list) in random_items:
        #"1833195": [{"answer": "computer", "confidence": 0.34137020439179955},
        k_candidates = qid_to_topkcands[str(question_id)][:k_cands]
        context = iid_to_capt[str(image_id)]
        candidates = construct_candidates(k_candidates)
        example_res += "Context: {}\n Question: {}\n Candidates: {}\n Answer: {}\n".format(context,question,candidates,find_most_ans(answer_list))
    return example_res

def get_k_answer_aware_examples_candidates(train_annotation_dict,annotation_dict, qid, iid_to_capt, qid_to_topkcands, qid_to_answer_aware_example, k, k_cands):
    example_res = ""
    # Choose k random key-value pairs
    # random_items = random.sample(list(annotation_dict.items()), k)
    similar_qids = get_similar_qids(qid, qid_to_answer_aware_example, k)
    similar_items = {}
    for similar_qid in similar_qids:
        similar_items[similar_qid] = train_annotation_dict[int(similar_qid)]
        
    for question_id, (image_id, question, answer_list) in similar_items.items():
        #"1833195": [{"answer": "computer", "confidence": 0.34137020439179955},
        k_candidates = qid_to_topkcands[str(question_id)][:k_cands]
        context = iid_to_capt[str(image_id)]
        candidates = construct_candidates(k_candidates)
        example_res += "Context: {}\n Question: {}\n Candidates: {}\n Answer: {}\n".format(context,question,candidates,find_most_ans(answer_list))
    return example_res

def get_k_question_aware_examples_candidates(train_annotation_dict,annotation_dict, qid, iid_to_capt, qid_to_topkcands, qid_to_question_aware_example, k, k_cands):
    example_res = ""
    # Choose k random key-value pairs
    # random_items = random.sample(list(annotation_dict.items()), k)
    similar_qids = get_similar_qids(qid, qid_to_question_aware_example, k)
    similar_items = {}
    for similar_qid in similar_qids:
        similar_items[similar_qid] = train_annotation_dict[int(similar_qid)]
        
    for question_id, (image_id, question, answer_list) in similar_items.items():
        #"1833195": [{"answer": "computer", "confidence": 0.34137020439179955},
        k_candidates = qid_to_topkcands[str(question_id)][:k_cands]
        context = iid_to_capt[str(image_id)]
        candidates = construct_candidates(k_candidates)
        example_res += "Context: {}\n Question: {}\n Candidates: {}\n Answer: {}\n".format(context,question,candidates,find_most_ans(answer_list))
    return example_res

def get_similar_qids(qid, qid_to_answer_aware_example, k=None):
    similar_qids = qid_to_answer_aware_example[str(qid)]
    if k is not None:
        similar_qids = similar_qids[:k]
    return similar_qids
    

def construct_prompt(annotation_dict, image_url_dict,train_annotation_dict):
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
        # cur_prompt = [prompt1, prompt2, prompt3]
        cur_prompt = [prompt2, prompt3]
        prompt_list.append(cur_prompt)
        question_id_list.append(question_id)
    return prompt_list, question_id_list


def construct_prompt_PICa(annotation_dict, image_url_dict,train_annotation_dict,iid_to_capt,k):
    """
    format:
    prompts = [
    "Instruction: please answer the question according to the above context. Use the image to answer.\n",
    "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg",
    "Question: What's on the picture? Answer: \n"
    ]
    """

    prompt_list = []
    question_id_list = []

    # prompt_context = construct_context(n_examples)

    prompt_head = "Instruction: please answer the question according to the above context. Use the image to answer.\n"
    #use same pica examples
    # PICa_examples = get_k_examples_PICa(annotation_dict, iid_to_capt,k)

    for question_id, (image_id, question, answer_list) in annotation_dict.items():
        #use different pica examples
        PICa_examples = get_k_examples_PICa(train_annotation_dict, iid_to_capt,k)
        context = iid_to_capt[str(image_id)]
        prompt2 = image_url_dict[image_id]
        prompt3 = "Context: {}\n Question: {} Answer: ".format(context,question)
        cur_prompt = [prompt_head, PICa_examples, prompt2, prompt3]
        # cur_prompt = [prompt2, prompt3]
        prompt_list.append(cur_prompt)
        question_id_list.append(question_id)
    return prompt_list, question_id_list

def construct_prompt_answer_aware(annotation_dict, image_url_dict,train_annotation_dict,iid_to_capt,qid_to_answer_aware_example,k):
    """
    format:
    prompts = [
    "Instruction: please answer the question according to the above context. Use the image to answer.\n",
    "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg",
    "Question: What's on the picture? Answer: \n"
    ]
    """

    prompt_list = []
    question_id_list = []

    # prompt_context = construct_context(n_examples)

    prompt_head = "Instruction: please answer the question according to the above context. Use the image to answer.\n"
    #use same pica examples
    # PICa_examples = get_k_examples_PICa(annotation_dict, iid_to_capt,k)

    for question_id, (image_id, question, answer_list) in annotation_dict.items():
        #use different pica examples
        answer_aware_examples = get_k_examples_answer_aware(train_annotation_dict,annotation_dict, question_id, qid_to_answer_aware_example,k)
        context = iid_to_capt[str(image_id)]
        prompt2 = image_url_dict[image_id]
        prompt3 = "Context: {}\n Question: {} Answer: ".format(context,question)
        cur_prompt = [prompt_head, answer_aware_examples, prompt2, prompt3]
        # cur_prompt = [prompt2, prompt3]
        prompt_list.append(cur_prompt)
        question_id_list.append(question_id)
    return prompt_list, question_id_list

def construct_prompt_question_aware(annotation_dict, image_url_dict,train_annotation_dict,iid_to_capt,qid_to_question_aware_example,k):
    """
    format:
    prompts = [
    "Instruction: please answer the question according to the above context. Use the image to answer.\n",
    "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg",
    "Question: What's on the picture? Answer: \n"
    ]
    """

    prompt_list = []
    question_id_list = []

    # prompt_context = construct_context(n_examples)

    prompt_head = "Instruction: please answer the question according to the above context. Use the image to answer.\n"
    #use same pica examples
    # PICa_examples = get_k_examples_PICa(annotation_dict, iid_to_capt,k)

    for question_id, (image_id, question, answer_list) in annotation_dict.items():
        #use different pica examples
        answer_aware_examples = get_k_examples_answer_aware(train_annotation_dict,annotation_dict, question_id, qid_to_answer_aware_example,k)
        context = iid_to_capt[str(image_id)]
        prompt2 = image_url_dict[image_id]
        prompt3 = "Context: {}\n Question: {} Answer: ".format(context,question)
        cur_prompt = [prompt_head, answer_aware_examples, prompt2, prompt3]
        # cur_prompt = [prompt2, prompt3]
        prompt_list.append(cur_prompt)
        question_id_list.append(question_id)
    return prompt_list, question_id_list

def construct_prompt_answer_aware_cands(train_annotation_dict,annotation_dict, image_url_dict,iid_to_capt,qid_to_topkcands,qid_to_answer_aware_example,k,k_cands):
    """
    format:
    prompts = [
    "Instruction: please answer the question according to the above context. Use the image to answer.\n",
    "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg",
    "Question: What's on the picture? Answer: \n"
    ]
    """

    prompt_list = []
    question_id_list = []

    # prompt_context = construct_context(n_examples)

    prompt_head = "Instruction: Please answer the question according to the context and answer candidates . Each answer candidate is associated with a confidence score within a bracket. The true answer may not be included in the candidates. Use the image to answer.\n"
    #use same pica examples
    # PICa_examples = get_k_examples_PICa(annotation_dict, iid_to_capt,k)

    for question_id, (image_id, question, answer_list) in annotation_dict.items():
        #use different pica examples
        k_candidats_examples = get_k_answer_aware_examples_candidates(train_annotation_dict,annotation_dict, question_id, iid_to_capt, qid_to_topkcands, qid_to_answer_aware_example, k, k_cands)
        context = iid_to_capt[str(image_id)]
        k_candidates = qid_to_topkcands[str(question_id)][:k_cands]
        candidates = construct_candidates(k_candidates)
        prompt2 = image_url_dict[image_id]
        prompt3 = "Context: {}\n Question: {}\n Candidates: {}\n Answer: ".format(context,question,candidates)
        cur_prompt = [prompt_head, k_candidats_examples, prompt2, prompt3]
        # cur_prompt = [prompt2, prompt3]
        prompt_list.append(cur_prompt)
        question_id_list.append(question_id)
    return prompt_list, question_id_list

def construct_prompt_question_aware_cands(train_annotation_dict,annotation_dict, image_url_dict,iid_to_capt,qid_to_topkcands,qid_to_question_aware_example,k,k_cands):
    """
    format:
    prompts = [
    "Instruction: please answer the question according to the above context. Use the image to answer.\n",
    "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg",
    "Question: What's on the picture? Answer: \n"
    ]
    """

    prompt_list = []
    question_id_list = []

    # prompt_context = construct_context(n_examples)

    prompt_head = "Instruction: Please answer the question according to the context and answer candidates . Each answer candidate is associated with a confidence score within a bracket. The true answer may not be included in the candidates. Use the image to answer.\n"
    #use same pica examples
    # PICa_examples = get_k_examples_PICa(annotation_dict, iid_to_capt,k)

    for question_id, (image_id, question, answer_list) in annotation_dict.items():
        #use different pica examples
        k_candidats_examples = get_k_question_aware_examples_candidates(train_annotation_dict,annotation_dict, question_id, iid_to_capt, qid_to_topkcands, qid_to_question_aware_example, k, k_cands)
        context = iid_to_capt[str(image_id)]
        k_candidates = qid_to_topkcands[str(question_id)][:k_cands]
        candidates = construct_candidates(k_candidates)
        prompt2 = image_url_dict[image_id]
        prompt3 = "Context: {}\n Question: {}\n Candidates: {}\n Answer: ".format(context,question,candidates)
        cur_prompt = [prompt_head, k_candidats_examples, prompt2, prompt3]
        # cur_prompt = [prompt2, prompt3]
        prompt_list.append(cur_prompt)
        question_id_list.append(question_id)
    return prompt_list, question_id_list

def construct_prompt_answer_cands(train_annotation_dict,annotation_dict, image_url_dict,iid_to_capt,qid_to_topkcands,k,k_cands):
    """
    format:
    prompts = [
    "Instruction: please answer the question according to the above context. Use the image to answer.\n",
    "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg",
    "Question: What's on the picture? Answer: \n"
    ]
    """

    prompt_list = []
    question_id_list = []

    # prompt_context = construct_context(n_examples)

    prompt_head = "Instruction: Please answer the question according to the context and answer candidates . Each answer candidate is associated with a confidence score within a bracket. The true answer may not be included in the candidates. Use the image to answer.\n"
    #use same pica examples
    # PICa_examples = get_k_examples_PICa(annotation_dict, iid_to_capt,k)

    for question_id, (image_id, question, answer_list) in annotation_dict.items():
        #use different pica examples
        k_candidats_examples = get_k_examples_candidates(train_annotation_dict,annotation_dict, iid_to_capt, qid_to_topkcands, k, k_cands)
        context = iid_to_capt[str(image_id)]
        k_candidates = qid_to_topkcands[str(question_id)][:k_cands]
        candidates = construct_candidates(k_candidates)
        prompt2 = image_url_dict[image_id]
        prompt3 = "Context: {}\n Question: {}\n Candidates: {}\n Answer: ".format(context,question,candidates)
        cur_prompt = [prompt_head, k_candidats_examples, prompt2, prompt3]
        # cur_prompt = [prompt2, prompt3]
        prompt_list.append(cur_prompt)
        question_id_list.append(question_id)
    return prompt_list, question_id_list



def vqa_acc(pred_answer, answer_list):

    answer_freq_dict = Counter(answer_list)
    num_human_ans = 0
    for answer, freq in answer_freq_dict.items():
        if answer.lower() in pred_answer.lower():
            num_human_ans += freq

    acc = min(num_human_ans/3,1)
    return acc


def evaluate(prompt_list, question_id_list, processor, model, annotation_dict):
    acc_list = []
    for i in tqdm(range(len(prompt_list))):
        prompts = prompt_list[i]
        question_id = question_id_list[i]
        inputs = processor(prompts, return_tensors="pt")
        inputs.to(device)

        generated_ids = model.generate(**inputs, max_length=150,)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        answer_list = annotation_dict[question_id][2]

        acc = vqa_acc(generated_text[0], answer_list)
        acc_list.append(acc)

        if (i + 1) % 100 == 0:
            print("current average accuracy is: ", np.mean(acc_list))
            # break
    print("final average accuracy is: ", np.mean(acc_list))

def extract_answer_from_predict(text,question,type_):
  # text = "Context: a bathroom with a glass shower door next to a toilet. Question: Name the type of plant this is? Answer: philodendron."
  """
   Context: a black motorcycle parked in a parking lot.
 Question: What sport can you use this for?
 Candidates: race (0.5339530181066084),motorcycle (0.40974219197095313),motocross (0.18935894667338782)
 Answer: motorcycle
Context: a man in a suit and tie standing in front of a building.

  """
  if type_ == "pica" or type_ == "answer_aware" or type_ == "question_aware":
    # Find the starting index of "Question:" and "."
    start_index = text.find(question)
    start_index += len(question)
    # start_index += len(" Answer:")
    start_index = text.find("Answer:",start_index)
    start_index += len("Answer:")
    
    # if end_index == -1:
    end_index1 = text.find("Question:",start_index)
    end_index2 = text.find("Context:",start_index)
    end_index3 = text.find(".", start_index)
    if end_index1 != -1 and end_index2 != -1 and end_index3!=-1:
        end_index = min(end_index1,end_index2,end_index3)
    elif end_index1 == -1 and ( end_index2 != -1 and end_index3!=-1):
        end_index = min(end_index2,end_index3)
    elif end_index2 == -1 and ( end_index1 != -1 and end_index3!=-1):
        end_index = min(end_index1,end_index3)
    elif end_index3 == -1 and ( end_index2 != -1 and end_index1!=-1):
        end_index = min(end_index2,end_index1)
    else:
        end_index = max(end_index1,end_index2,end_index3)
    if end_index == -1:
      end_index = len(text)
    if start_index != -1 and end_index != -1:
        cleaned_answer = text[start_index:end_index].strip()
        return cleaned_answer
    else:
        print("Text not found.")
  if type_ == 'answer_candidates' or type_ == "answer_aware_candidates":
    start_index = text.find(question)
    start_index += len(question)
    # start_index += len(" Answer:")
    start_index = text.find("Answer: ", start_index)
    start_index += len("Answer: ")
    end_index = text.find("Context", start_index)
    if end_index == -1:
      end_index = len(text)
    if start_index != -1 and end_index != -1:
        cleaned_answer = text[start_index:end_index].strip()
        return cleaned_answer
    else:
        print("Text not found.")


def evaluate_batch(prompt_list, question_id_list, processor, model, question_dict, annotation_dict, result_path, type_ , batch_size=1):
    tokenizer = processor.tokenizer
    # eos_token = ["</s>"]
    # eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    eos_token_id = [2, 13]

    # f = open(result_path, "w")
    # f.write("question_id,image_id,question,prediction,answer_list\n")
    result = pd.DataFrame(columns=["question_id", "image_id", "question", "prediction", "prediction_clean", "answer_list"])

    acc_list = []
    for i in range((len(prompt_list) + 1) // batch_size):
        prompts = prompt_list[i * batch_size: (i+1) * batch_size]
        question_ids = question_id_list[i * batch_size: (i+1) * batch_size]
        inputs = None
        while not inputs:
            try:
                inputs = processor(prompts, return_tensors="pt").to(device)   
            except Exception as e:
                print(e)
                time.sleep(1)
        # inputs = processor(prompts, return_tensors="pt").to(device)

        generated_ids = model.generate(**inputs, eos_token_id=eos_token_id, max_new_tokens=100,)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for j in range(len(prompts)):

            qid = question_ids[j]
            answer_list = annotation_dict[qid][2]
            # print("-------generated text--------")
            # print(generated_text[j])

            pred_answer = generated_text[j].split("Answer:")[1]
            pred_answer_clean = extract_answer_from_predict(generated_text[j],question_dict[qid][1],type_)#pred_answer.split("\n")[0].split(". ")[0]  # get the result after Answer:
            # print("-------cleaned answer--------")
            # print(pred_answer_clean)
            acc = vqa_acc(pred_answer_clean, answer_list)
            acc_list.append(acc)

            # tmp = "%d,%d,%s,%s,%s\n"%(qid, question_dict[qid][0], question_dict[qid][1], pred_answer, "|".join(answer_list))
            # f.write(tmp)
            result.loc[len(result.index)] = [qid, question_dict[qid][0], question_dict[qid][1], pred_answer, pred_answer_clean, "|".join(answer_list)]

        if (i + 1) * batch_size % 100 < batch_size:
            print("current average accuracy is: ", np.mean(acc_list))
            result.to_csv(result_path, index=False)
        # if (i + 1) % 10 == 0:
        #     print("current average accuracy is: ", np.mean(acc_list))
        #     result.to_csv(result_path, index=False)
            # break
    print("final average accuracy is: ", np.mean(acc_list))
    result.to_csv(result_path, index=False)
    return np.mean(acc_list)



if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    print("--------------   import done   ---------------")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    #checkpoint = "HuggingFaceM4/tiny-random-idefics"

    model, processor = load_model(checkpoint)
    model.load_adapter("yujiaw2/idefics-9b-instruct-okvqa-v4")

    print("--------------   model loaded   ---------------")

    val_question_path =  'datasets/okvqa/OpenEnded_mscoco_val2014_questions.json'
    val_annotation_path =  'datasets/okvqa/mscoco_val2014_annotations.json'
    train_annotation_path =  'datasets/okvqa/mscoco_train2014_annotations.json'
    train_question_path =  'datasets/okvqa/OpenEnded_mscoco_train2014_questions.json'
    caption_path = 'assets/captions_okvqa.json'
    candidates_path = 'assets/candidates_okvqa.json'
    answer_aware_path = 'assets/answer_aware_examples_okvqa.json'
    question_aware_path = "assets/similar_questions_okvqa.json"
    
    train_question_dict = process_question_file(train_question_path)
    train_question_dict = process_question_file(train_question_path)
    val_question_dict = process_question_file(val_question_path)
    val_question_dict = process_question_file(val_question_path)
    print("--------------   question loaded   ---------------")

    train_annotation_dict = process_annotation_file(train_annotation_path, train_question_dict)
    train_image_url_dict = image_to_url(train_question_dict)
    val_annotation_dict = process_annotation_file(val_annotation_path, val_question_dict)
    val_image_url_dict = image_to_url(val_question_dict)
    iid_to_capt = process_caption(caption_path)
    qid_to_topkcands = process_caption(candidates_path)
    qid_to_answer_aware_example = process_caption(answer_aware_path)
    qid_to_question_aware_example = process_caption(question_aware_path)
 
    print("--------------   info loaded   ---------------")

    
    prompt_type = sys.argv[1]

    k = int(sys.argv[2])
    prompt_list, question_id_list, result_path = [],[],''
    if prompt_type == "pica":
        prompt_list, question_id_list = construct_prompt_PICa(val_annotation_dict, val_image_url_dict,train_annotation_dict,iid_to_capt,k)
        # construct_prompt(val_annotation_dict, val_image_url_dict,train_annotation_dict)
        result_path = "outputs/{}_{}result.csv".format(prompt_type,k)
        print(f'--------- prompt type is {prompt_type}, k is {k}')
        
    if prompt_type == "answer_aware":
        prompt_list, question_id_list = construct_prompt_answer_aware(val_annotation_dict, val_image_url_dict,train_annotation_dict,iid_to_capt,qid_to_answer_aware_example,k)
        # construct_prompt(val_annotation_dict, val_image_url_dict,train_annotation_dict)
        result_path = "outputs/{}_{}result.csv".format(prompt_type,k)
        print(f'--------- prompt type is {prompt_type}, k is {k}')
    
    if prompt_type == "question_aware":
        prompt_list, question_id_list = construct_prompt_question_aware(val_annotation_dict, val_image_url_dict,train_annotation_dict,iid_to_capt,qid_to_question_aware_example,k)
        # construct_prompt(val_annotation_dict, val_image_url_dict,train_annotation_dict)
        result_path = "outputs/{}_{}result.csv".format(prompt_type,k)
        print(f'--------- prompt type is {prompt_type}, k is {k}')
        
    if prompt_type == "answer_candidates":
        k_cands = int(sys.argv[3])
        prompt_list, question_id_list = construct_prompt_answer_cands(train_annotation_dict,val_annotation_dict, val_image_url_dict,iid_to_capt,qid_to_topkcands,k,k_cands)
        result_path = "outputs/{}_{}_{}cands_result.csv".format(prompt_type,k,k_cands)
        print(f'--------- prompt type is {prompt_type}, k is {k}, k_cands is {k_cands}')
    
    if prompt_type == "answer_aware_candidates":
        k_cands = int(sys.argv[3])
        prompt_list, question_id_list = construct_prompt_answer_aware_cands(train_annotation_dict,val_annotation_dict, val_image_url_dict,iid_to_capt,qid_to_topkcands,qid_to_answer_aware_example,k,k_cands)
        result_path = "outputs/{}_{}_{}cands_result.csv".format(prompt_type,k,k_cands)
        print(f'--------- prompt type is {prompt_type}, k is {k}, k_cands is {k_cands}')
    
    if prompt_type == "question_aware_candidates":
        k_cands = int(sys.argv[3])
        prompt_list, question_id_list = construct_prompt_question_aware_cands(train_annotation_dict,val_annotation_dict, val_image_url_dict,iid_to_capt,qid_to_topkcands,qid_to_question_aware_example,k,k_cands)
        result_path = "outputs/{}_{}_{}cands_result.csv".format(prompt_type,k,k_cands)
        print(f'--------- prompt type is {prompt_type}, k is {k}, k_cands is {k_cands}')
        
    
    
   
    # prompt_list, question_id_list = construct_prompt(val_annotation_dict, val_image_url_dict,train_annotation_dict)
    print("--------------   prompt done   ---------------")
    
    

    acc = evaluate_batch(prompt_list, question_id_list, processor, model, val_question_dict, val_annotation_dict, result_path=result_path,type_=prompt_type, batch_size=16)
    print("------------- final accuracy for this prompt setting is {} ----------".format(acc))
    torch.cuda.empty_cache()
