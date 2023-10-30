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
import time

def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
      """Pad a sequence to the desired max length."""
      if len(sequence) >= max_length:
          return sequence
      return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])

def format_text(inp, model, conv):
    # first message
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    # else:
    #     # later messages
    #     conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt



def get_processed_tokens_batch( batch_text, image_paths, image_processor, tokenizer, model):
      prompt = [format_text(text + " Answer concisely, in only one word.", model, conv_templates[args.conv_mode].copy()) for text in batch_text]
      images = [load_image('/data/user_data/naveensu/CLEVR_v1.0/images/val/' + image_path) for image_path in image_paths]

      batch_input_ids = [
          tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompt
      ]

      # Determine the maximum length of input_ids in the batch
      max_len = max([len(seq) for seq in batch_input_ids])
      # Pad each sequence in input_ids to the max_len
      padded_input_ids = [pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in batch_input_ids]
      batch_input_ids = torch.stack(padded_input_ids).cuda()

      batch_image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().cuda()
      
      return batch_image_tensor, batch_input_ids

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
    tokenizer.padding_side = "left"

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



    batch_size = 32
    for i in range(0, len(questions_list), batch_size):
        print("Batch ", int(i/batch_size) + 1)
        image_filenames = [j['image_filename'] for j in questions_list[i: i + batch_size]]
        questions = [j['question'] for j in questions_list[i: i + batch_size]]

        batch_image_tensor, batch_input_ids = get_processed_tokens_batch(questions, image_filenames, image_processor, tokenizer, model)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, batch_input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                batch_input_ids,
                images=batch_image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=128,
                # streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria for _ in range(batch_size)])

        outputs = tokenizer.batch_decode(output_ids[:, batch_input_ids.shape[1]:])
        outputs = [out.strip() for out in outputs]
        outputs = [output.split(stop_str)[0] for output in outputs]
        file_dump['question'] += questions
        file_dump['image_filename'] += image_filenames
        file_dump['ground_truth_ans'] += [j['answer'] for j in questions_list[i: i + batch_size]]
        file_dump['question_type'] += [','.join(i['function'] for i in question['program']) for question in questions_list[i: i + batch_size]]
        file_dump['predicted_ans'] += outputs
        file_dump['correct'] += [1 for _ in range(batch_size)]
        file_dump['question_family_index'] += [j['question_family_index'] for j in questions_list[i: i + batch_size]]
        if (i/batch_size) % 20 == 0:
            pd.DataFrame(file_dump).to_csv(args.eval_file + '.csv', index = False)
        if args.debug:
            print(questions)
            print(outputs)
        
    pd.DataFrame(file_dump).to_csv(args.eval_file + '.csv', index = False)

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
    parser.add_argument("--eval-file", type = str, default = "temp-eval")
    args = parser.parse_args()
    main(args)

