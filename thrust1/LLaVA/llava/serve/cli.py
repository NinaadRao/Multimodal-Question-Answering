import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import json
import pandas as pd

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    im_path = '/data/user_data/naveensu/CLEVR_v1.0/images/val/'
    questions_file =open('/data/user_data/naveensu/CLEVR_v1.0/questions/CLEVR_val_questions.json')
    questions_list = json.load(questions_file)['questions']
    questions_file.close()
    file_dump = {'question' : [], 'image_filename' : [], 'ground_truth_ans' : [], 'question_type' : [],'predicted_ans' : [], 'correct': [], 'question_family_index': []}
    pd.DataFrame(file_dump).to_csv('outputs-eval.csv', index = False)



    # while True:
    prompts = []
    batch_size = 16
    print(len(questions_list))
    for i, question in enumerate(questions_list):
        if (i + 1) % batch_size == 0:
            
            input_ids, attention_mask = tokenizer_image_token(prompts, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.cuda()
            # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            # keywords = [stop_str]
            print(input_ids[:1, ].size())
            # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids[:1, ])
            # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    # streamer=streamer,
                    use_cache=True,
                    # stopping_criteria=[stopping_criteria], 
                    attention_mask = attention_mask)
            print(output_ids.size())
            outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:])
            print(outputs)
            # file_dump['question'].append(question['question'])
            # file_dump['image_filename'].append(question['image_filename'])
            # file_dump['ground_truth_ans'].append(question['answer'])
            # file_dump['question_type'].append(','.join(i['function'] for i in question['program']))
            # file_dump['predicted_ans'].append(outputs)
            # file_dump['correct'].append(1)
            # file_dump['question_family_index'].append(question['question_family_index'])
            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

            prompts = []
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
            df = pd.read_csv('outputs-eval.csv')
            pd.concat([df, pd.DataFrame(file_dump)])

            
        conv = conv_templates[args.conv_mode].copy()
        
        im_file = question['image_filename']
        print(i + 1, im_file)

        inp = question['question']
        image = load_image(im_path + im_file)
        if (i + 1) % batch_size:
            image_tensor = torch.cat((image_tensor, image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()), 0)

        print(image_tensor.size())
        
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        # else:
        #     # later messages
        #     conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)
        
        
    pd.DataFrame(file_dump).to_csv('outputs-eval.csv', index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
