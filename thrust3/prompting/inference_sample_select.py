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
from vqaEval import VQAEval
import random
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
import time

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


def build_vector_db(embed_data, path, name):
    # Persists changes to disk
    client = QdrantClient(path=path)

    # create collection
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
    )
    # insert data
    client.upsert(
        collection_name=name,
        points=[
            PointStruct(
                id = int(row["question_id"]),
                vector = row.values[1:]
            )
            for idx, row in embed_data.iterrows()
        ]
    )
    return client


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


def construct_prompt_few_shots_similar(annotation_dict, image_url_dict, annotation_dict_train, question_ids_train, embed_model, kNN=None, client=None, name="", shots=4):
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
    similar_examples = {}

    prompt1 = "Instruction: provide a short answer to the question. Use the image to answer.\n"

    for question_id, (image_id, question, answer_list) in tqdm(annotation_dict.items()):
        prompt2 = image_url_dict[image_id]
        prompt3 = "Question: {} Answer: ".format(question)
        cur_prompt = [prompt1, prompt2, prompt3]
        # cur_prompt = [prompt2, prompt3]

        #### mode 1: search kNN now
        if not kNN:
            # test sample embedding
            inputs = None
            while not inputs:
                try:
                    inputs = processor(cur_prompt, return_tensors="pt").to(device)
                except Exception as e:
                    print(e)
                    time.sleep(1)
        
            output = embed_model.forward(**inputs, output_hidden_states=True)
            embed = output["hidden_states"][-1][0][-1]

            few_shots, qids = choose_examples_train_similar(annotation_dict_train, question_ids_train, embed=embed.cpu().detach().numpy(),
                                                            client=client, name=name, shots=shots)
            similar_examples[question_id] = tuple(qids)
        
        #### mode 2: use existing results
        else:
            few_shots, qids = choose_examples_train_similar(annotation_dict_train, question_ids_train, kNN=kNN, question_id=question_id, shots=shots)

        prompt_list.append(few_shots + cur_prompt)
        question_id_list.append(question_id)      
    
    ### save kNN
    if not kNN:
        with open('data/similar_examples_%d.json'%shots, 'w') as f:
            json.dump(similar_examples, f)
    return prompt_list, question_id_list


def choose_examples_train_similar(annotation_dict_train, question_ids_train, embed=None, kNN=None, client=None, name="", question_id=None, shots=4):
    # qids = random.sample(question_ids_train, shots)

    #### mode 1: search kNN now
    if not kNN:
        # search
        hits = client.search(
            collection_name=name,
            query_vector=embed,
            limit=shots  # Return 4 closest points
        )
        qids = [item.id for item in hits]
    #### mode 2: use existing results
    else:
        qids = kNN[str(question_id)]

    #### design 4: plain template
    few_shots = []
    prompt1 = "Instruction: provide a short answer to the question. Use the image to answer.\n"
    for qid in qids:
        image_id, question, answer_list = annotation_dict_train[qid]
        prompt2 = "https://okvqa11632.s3.amazonaws.com/train2014/COCO_train2014_"+f"{image_id:012d}"+".jpg"
        prompt3 = "Question: {} Answer: {}.\n\n".format(question, random.choice(answer_list))
        few_shots = few_shots + [prompt1, prompt2, prompt3]
    return few_shots, qids

    #### design 6: possible answers
    # few_shots = []
    # prompt1 = "Instruction: Based on the image, provide possible answers to the question. Choose the most logical answer as the final answer.\n"
    # for qid in qids:
    #     image_id, question, answer_list = annotation_dict_train[qid]
    #     answer_dict = Counter(answer_list)

    #     prompt2 = "https://okvqa11632.s3.amazonaws.com/train2014/COCO_train2014_"+f"{image_id:012d}"+".jpg"
    #     prompt3 = "Question: {}\n".format(question)
    #     prompt4 = "Possible Answers: [{}]\n".format(", ".join(list(answer_dict.keys())))
    #     prompt5 = "Final Answer: {}\n\n".format(max(answer_dict, key=answer_dict.get))
    #     few_shots = few_shots + [prompt1, prompt2, prompt3, prompt4, prompt5]
    # return few_shots, qids


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

        generated_ids = model.generate(**inputs, max_length=100,)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        answer_list = annotation_dict[question_id][2]

        acc = vqa_acc(generated_text[0], answer_list)
        acc_list.append(acc)

        if (i + 1) % 100 == 0:
            print("current average accuracy is: ", np.mean(acc_list))
            # break
    print("final average accuracy is: ", np.mean(acc_list))


def evaluate_batch(vqa_eval, prompt_list, question_id_list, processor, model, question_dict, annotation_dict, result_path, batch_size=1):
    tokenizer = processor.tokenizer
    # eos_token = ["</s>"]
    # eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    # eos_token_id = [2, 13]
    eos_token_id = [2]

    # f = open(result_path, "w")
    # f.write("question_id,image_id,question,prediction,answer_list\n")
    result = pd.DataFrame(columns=["question_id", "image_id", "question", "prediction", "prediction_clean", "answer_list", "acc"])
    # result = pd.DataFrame(columns=["question_id", "image_id", "question", "possible_answers", "prediction", "prediction_clean", "answer_list", "acc"])

    acc_list1 = []
    acc_list2 = []
    for i in tqdm(range((len(prompt_list) - 1) // batch_size + 1)):
        prompts = prompt_list[i * batch_size: (i+1) * batch_size]
        question_ids = question_id_list[i * batch_size: (i+1) * batch_size]

        inputs = None
        while not inputs:
            try:
                inputs = processor(prompts, return_tensors="pt").to(device)   
            except Exception as e:
                print(e)
                time.sleep(1)
                
        # import pdb; pdb.set_trace()
        generated_ids = model.generate(**inputs, eos_token_id=eos_token_id, max_new_tokens=20,)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for j in range(len(prompts)):
            qid = question_ids[j]
            answer_list = annotation_dict[qid][2]

            ## get the result after the last Answer:
            # possible_answers = generated_text[j].split("Possible Answers:")[-1].split("\n")[0] 
            pred_answer = generated_text[j].split("Answer:")[-1].split("Question:")[0].split("\n")[0] 
       
            # acc = vqa_acc(pred_answer, answer_list)
            acc1, pred_answer_clean = vqa_eval.evaluate(pred_answer, answer_list)
            acc_list1.append(acc1)

            acc2, pred_answer_clean = vqa_eval.evaluate(pred_answer, answer_list, exact_match=True)
            acc_list2.append(acc2)

            # tmp = "%d,%d,%s,%s,%s\n"%(qid, question_dict[qid][0], question_dict[qid][1], pred_answer, "|".join(answer_list))
            # f.write(tmp)
            # result.loc[len(result.index)] = [qid, question_dict[qid][0], question_dict[qid][1], possible_answers, pred_answer, pred_answer_clean, "|".join(answer_list), acc1]
            result.loc[len(result.index)] = [qid, question_dict[qid][0], question_dict[qid][1], pred_answer, pred_answer_clean, "|".join(answer_list), acc1]

        if (i + 1) * batch_size % 100 < batch_size:
            print("current substring accuracy is: %.6f, exact match accuracy is: %.6f" %(np.mean(acc_list1), np.mean(acc_list2)))
            result.to_csv(result_path, index=False)
            # break
    print("final substring accuracy is: %.6f, exact match accuracy is: %.6f" %(np.mean(acc_list1), np.mean(acc_list2)))
    result.to_csv(result_path, index=False)


if __name__ == "__main__":
    
    print("--------------   import done   ---------------")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    #checkpoint = "HuggingFaceM4/tiny-random-idefics"
    peft_model_id = "yujiaw2/idefics-9b-instruct-okvqa-v4"
    collection_name = "okvqa_collection"

    if_search_kNN = False
    shots = 16
    finetuned = True

    model, processor = load_model(checkpoint)
    # model.load_adapter("yujiaw2/idefics-9b-instruct-okvqa-v4")  # for embedding (not used)

    print("--------------   model loaded   ---------------")

    question_path =  'data/OpenEnded_mscoco_val2014_questions.json'
    annotation_path =  'data/mscoco_val2014_annotations.json'
    question_dict, _ = process_question_file(question_path)
    annotation_dict = process_annotation_file(annotation_path, question_dict)
    image_url_dict = image_to_url(question_dict)

    question_path_train =  'data/OpenEnded_mscoco_train2014_questions.json'
    annotation_path_train =  'data/mscoco_train2014_annotations.json'
    question_dict_train, question_ids_train = process_question_file(question_path_train)
    annotation_dict_train = process_annotation_file(annotation_path_train, question_dict_train)
    print("--------------   question & info loaded   ---------------")

    # import pdb; pdb.set_trace()
    

    if if_search_kNN:
        #### mode 1: use vector db to retrieve similar examples
        model = PeftModel.from_pretrained(model, model_id=peft_model_id, adapter_name="embed")
        lora_config = LoraConfig(target_modules=["q_proj", "k_proj"], init_lora_weights=False)
        model.add_adapter(peft_config=lora_config, adapter_name="base")
        model.set_adapter("embed")

        embed_data = pd.read_csv("data/training_embed.csv")
        client = build_vector_db(embed_data, path="./db", name=collection_name)
    else:
        #### mode 2: load from saved file
        with open("data/similar_examples_%d.json" %shots, 'r') as json_file:
            kNN = json.load(json_file)

        if finetuned:
            # inference with finetuned model
            model = PeftModel.from_pretrained(model, model_id=peft_model_id, adapter_name="embed")


    print("--------------   vector db loaded   ---------------")
    ## zero-shot
    # prompt_list, question_id_list = construct_prompt(annotation_dict, image_url_dict) 
    
    # import pdb; pdb.set_trace()
    ## few-shots similar
    if if_search_kNN:
        prompt_list, question_id_list = construct_prompt_few_shots_similar(annotation_dict, image_url_dict, annotation_dict_train, 
                                                                            question_ids_train, model, client=client, name=collection_name, shots=shots)
    else:
        prompt_list, question_id_list = construct_prompt_few_shots_similar(annotation_dict, image_url_dict, annotation_dict_train, 
                                                                            question_ids_train, model, kNN=kNN, shots=shots)
    
    ## switch to base model (needed?)
    # model.set_adapter("base")

    print("--------------   prompt done   ---------------")

    vqa_eval = VQAEval()

    import pdb; pdb.set_trace()
    evaluate_batch(vqa_eval, prompt_list, question_id_list, processor, model, question_dict, annotation_dict, result_path="./result_16_shots_similar.csv", batch_size=12)

    import pdb; pdb.set_trace()