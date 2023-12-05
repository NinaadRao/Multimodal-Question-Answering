from transformers import BertTokenizer, BertModel
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity
# from generate_prompt import *

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'  # Example: You can choose other variations such as 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

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

# return a list of answers to a question
def process_answers(answer_list):
    res_answer = []
    for p in answer_list:
        res_answer.append(p['answer'])
    return res_answer
#process files

train_annotation_path =  'datasets/okvqa/mscoco_train2014_annotations.json'
train_question_path =  'datasets/okvqa/OpenEnded_mscoco_train2014_questions.json'


train_question_dict = process_question_file(train_question_path)

print("--------------   question loaded   ---------------")

train_annotation_dict = process_annotation_file(train_annotation_path, train_question_dict)

print("--------------   info loaded   ---------------")
# format: question_id: (image_id, question)



# Your list of questions and their question_ids
# questions = [
#     {"question_id": 1, "question_text": "What is the capital of France?"},
#     {"question_id": 2, "question_text": "Who is the president of the USA?"},
#     # Add more questions...
# ]

# Tokenize and obtain BERT embeddings for each question
question_embeddings = []
question_ids = []
for question_id,(image_id, question_text) in train_question_dict.items():
    question_ids.append(question_id)
    input_ids = tokenizer.encode(question_text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(input_ids)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
    question_embeddings.append(embeddings)

print("--------------   embedding loaded   ---------------")
similar_qids = {}
# Calculate cosine similarity between question embeddings
for i in range(len(question_ids)):
    current_embedding = question_embeddings[i].reshape(1, -1)
    similarity_scores = cosine_similarity(current_embedding, torch.stack(question_embeddings)).flatten()

    # Create a list of (question_id, similarity_score) tuples
    similar_questions = [
        (question_ids[j], similarity_scores[j])
        for j in similarity_scores.argsort()[::-1][1:101]  # Exclude self, get top 100
    ]
    similar_questions_ids = [
        question_ids[j] for j in similarity_scores.argsort()[::-1][1:101]  # Exclude self, get top 100
    ]
    similar_qids[question_ids[i]] = similar_questions_ids

print("--------------   similarity calculated   ---------------")
with open('/alicia_prompt/prompting/assets/similar_questions_okvqa.json', 'w') as f:
    json.dump(similar_qids, f)