import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
from PIL import Image
# import boto3
import json
import pandas as pd
from peft import PeftModel
import time
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
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(model, 'clevr-100_run-1')

processor = AutoProcessor.from_pretrained(checkpoint)


### some variables are purposely kept blank

# session = boto3.Session(
#     aws_access_key_id="",
#     aws_secret_access_key=""
# )

# s3 = session.client('s3')
# bucket_name = ""
# key = ""

# obj = s3.get_object(Bucket=bucket_name, Key=key)
# file_content = json.loads(obj['Body'].read().decode('utf-8'))
file = open("/data/user_data/naveensu/vqa/vqacp_v2_test_annotations.json")
annot_file_content = json.loads(file.read())

file = open("/data/user_data/naveensu/vqa/vqacp_v2_test_questions.json")
ques_file_content = json.loads(file.read())


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
               # 'question_family_index': [],
               'coco_split': []
              }

base_url = "/data/user_data/naveensu/vqa/"

prompts = []
ground_truth_answers = []
for (item1, item2) in zip(annot_file_content, ques_file_content):
    # start = time.time()
    total_questions += 1
    question = item2['question']
    img_filename = item2['image_id']
    coco_split = item2['coco_split']
    # print(item1)
    # print(item2)
    zeroes = '0' * (12 - len(str(img_filename)))
    image_object_url = base_url + str(coco_split) + '/' + 'COCO_' + coco_split + '_' + zeroes +  str(img_filename) + '.jpg'
    ground_truth_answer = item1['multiple_choice_answer']
    ground_truth_answers.append(ground_truth_answer)

    img = Image.open(image_object_url)

    prompts.append([img, "Question: " + question  + " Please answer in exactly one word. " + "Answer: "])

    # types = ''
    # for p in item['program']:
    #     if p['function'] == 'scene':
    #         continue
    #     types = types + p['function'] + ", "

    output_dict['question'].append(question)
    output_dict['image_filename'].append(img_filename)
    output_dict['ground_truth_ans'].append(ground_truth_answer)
    # output_dict['question_family_index'].append(item['question_family_index'])
    output_dict['question_type'].append(item1['question_type'])
    output_dict['coco_split'].append(coco_split)


    if total_questions % 128 == 0:

        predicted_answer = generate_output_answer(prompts)

        for ans, ground_truth in zip(predicted_answer, ground_truth_answers):
            # print(ans)
            
            ans = ans.split("Answer: ")
            if len(ans) > 1:
                ans = ans[1]

            output_dict['predicted_ans'].append(ans)

            if evaluate_result(ans, ground_truth):
                total_correct += 1
                output_dict['correct'].append(1)
            else:
                output_dict['correct'].append(0)

        print(f"Completed {total_questions}")

        prompts = []
        ground_truth_answers = []
        pd.DataFrame(output_dict).to_csv('output-vqa-ft1-2.csv', index = False)

    # print(time.time() - start)

for ans, ground_truth in zip(predicted_answer, ground_truth_answers):
    # print(ans)
    ans = ans.split("Answer: ")
    if len(ans) > 1:
        ans = ans[1]
    output_dict['predicted_ans'].append(ans)
    # print(ans)

    if evaluate_result(str(ans), str(ground_truth)):
        total_correct += 1
        output_dict['correct'].append(1)
    else:
        output_dict['correct'].append(0)

print(f"Completed {total_questions}")

df = pd.DataFrame(output_dict)
df.to_csv("output-vqa-ft1-2.csv", index = False)
#     if total_questions % 128 == 0:

#         predicted_answer = generate_output_answer(prompts)

#         for ans, ground_truth in zip(predicted_answer, ground_truth_answers):
#             # print(ans)
#             output_dict['predicted_ans'].append(ans)

#             if evaluate_result(ans, ground_truth):
#                 total_correct += 1
#                 output_dict['correct'].append(1)
#             else:
#                 output_dict['correct'].append(0)

#         print(f"Completed {total_questions}")

#         prompts = []

#         df = pd.DataFrame(output_dict)
#         df.to_csv("output.csv")

# predicted_answer = generate_output_answer(prompts)

# for ans, ground_truth in zip(predicted_answer, ground_truth_answers):
#     # print(ans)
#     ans = ans.split("Assistant: ")[1]
#     output_dict['predicted_ans'].append(ans)
#     # print(ans)

#     if evaluate_result(ans, ground_truth):
#         total_correct += 1
#         output_dict['correct'].append(1)
#     else:
#         output_dict['correct'].append(0)

# print(f"Completed {total_questions}")

# # prompts = []

# df = pd.DataFrame(output_dict)
# df.to_csv("output.csv")

