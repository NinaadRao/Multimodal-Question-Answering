from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompts, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    input_ids_final = None
    for prompt in prompts:
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                if input_ids_final is None:
                    input_ids_final = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                    attention_mask = torch.ones(input_ids_final.size())
                    input_ids_final = torch.nn.functional.pad(input_ids_final, (0, 512 - len(input_ids)), "constant", tokenizer.eos_token_id)
                    attention_mask = torch.nn.functional.pad(attention_mask, (0, 512 - len(input_ids)), "constant", 0)
                    
                    print(input_ids_final.size())
                else:
                    temp = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                    temp_attention_mask = torch.ones(temp.size())
                    temp = torch.nn.functional.pad(temp, (0, 512 - len(input_ids)), "constant", tokenizer.eos_token_id)
                    temp_attention_mask = torch.nn.functional.pad(temp_attention_mask, (0, 512 - len(input_ids)), "constant", 0)


                    input_ids_final = torch.cat((input_ids_final, temp), 0)
                    attention_mask = torch.cat((attention_mask, temp_attention_mask), 0)
                    print(input_ids_final.size())

    return input_ids_final, attention_mask


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        return False
