import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import  json
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args



def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True



# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')



#answer
#reset
#gradio_reset

#upload the image
with open('/workspace/CLEVR-dataset/questions/CLEVR_val_questions.json','r') as fp:
    df = json.load(fp)
generated_answers = []

file_name = 'val_output_minigpt.tsv'

with open(file_name,'w') as fp:
    fp.write("question\timage_filename\tground_truth_ans\tquestion_type\tpredicted_ans\tcorrect\tquestion_family_index\n")
chat_state = CONV_VISION.copy()
for i in tqdm(range(len(df['questions']))):
    question = df['questions'][i]['question']
    answer = df['questions'][i]['answer']
    image = df['questions'][i]['image_filename']
    question_family_index = str(df['questions'][i]['question_family_index'])
    correct = 'yes'
    question_type = []
    for j in df['questions'][i]['program']:
        question_type.append(j['function'])
    question_type = ','.join(question_type)
    
    
    


    
    img_list = []
    llm_message = chat.upload_img('/workspace/CLEVR-dataset/images/val/'+image, chat_state, img_list)

    #ask

    chat.ask(question, chat_state)

    #answer

    llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=1,
                                  temperature=1,
                                  max_new_tokens=300,
                                  max_length=1500)[0]
    chat_state.messages = []
    generated_answers.append('\t'.join([question,image,answer,question_type,llm_message[0],'yes',question_family_index]))


    # append results to output file
    if len(generated_answers)%10==0:
        with open(file_name,'a') as fp:
            fp.write('\n'.join(generated_answers))
            fp.write('\n')
            generated_answers = []
            break


