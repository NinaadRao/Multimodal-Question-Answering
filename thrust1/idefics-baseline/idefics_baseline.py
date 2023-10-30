import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image
import boto3
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)

### some variables are purposely kept blank

session = boto3.Session(
    aws_access_key_id="",
    aws_secret_access_key=""
)

s3 = session.client('s3')
bucket_name = ""
key = ""

obj = s3.get_object(Bucket=bucket_name, Key=key)
file_content = json.loads(obj['Body'].read().decode('utf-8'))

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
               'question_type': [],
               'predicted_ans': [], 
               'correct': [],
               'question_family_index': []
              }

base_url = "/workspace/CLEVR-dataset/images/"

prompts = []
ground_truth_answers = []
for item in file_content['questions']:
    total_questions += 1

    question = item['question']
    img_filename = item['image_filename']
    img_split = item['split']
    image_object_url = base_url + img_split + '/' + img_filename
    ground_truth_answer = item['answer']
    ground_truth_answers.append(ground_truth_answer)

    img = Image.open(image_object_url)

    prompts.append(["User: " + question, img, "<end_of_utterance>"])

    types = ''
    for p in item['program']:
        if p['function'] == 'scene':
            continue
        types = types + p['function'] + ", "

    output_dict['question'].append(question)
    output_dict['image_filename'].append(img_filename)
    output_dict['ground_truth_ans'].append(ground_truth_answer)
    output_dict['question_family_index'].append(item['question_family_index'])
    output_dict['question_type'].append(types)

    if total_questions % 128 == 0:

        predicted_answer = generate_output_answer(prompts)

        for ans, ground_truth in zip(predicted_answer, ground_truth_answers):
            # print(ans)
            ans = ans.split("Assistant: ")[1]
            output_dict['predicted_ans'].append(ans)

            if evaluate_result(str(ans), str(ground_truth)):
                total_correct += 1
                output_dict['correct'].append(1)
            else:
                output_dict['correct'].append(0)

        print(f"Completed {total_questions}")

        prompts = []
        
predicted_answer = generate_output_answer(prompts)

for ans, ground_truth in zip(predicted_answer, ground_truth_answers):
    # print(ans)
    ans = ans.split("Assistant: ")[1]
    output_dict['predicted_ans'].append(ans)
    # print(ans)

    if evaluate_result(str(ans), str(ground_truth)):
        total_correct += 1
        output_dict['correct'].append(1)
    else:
        output_dict['correct'].append(0)

print(f"Completed {total_questions}")

df = pd.DataFrame(output_dict)
df.to_csv("output.csv", index = False)

### separately process the last batch
