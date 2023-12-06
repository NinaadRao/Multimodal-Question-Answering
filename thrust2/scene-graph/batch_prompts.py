import json
import os
from PIL import Image

def batch_prompts_imgwise(qfile, save_path, sg_file=None):
  prompts = {0:[]}
  qid_batched = {0:[]}
  batchsize = 64
  b_idx = 0

  with open(qfile,'r') as f:
    qdata = json.load(f)

  if sg_file:
    with open(sg_file, 'r') as f:
      sg_data = json.load(f)

  count_no_sg = 0

  q_data_img = dict()

  for qid in qdata:
    question = qdata[qid]['question']
    answer = qdata[qid]['answer']
    sg = None

    if sg_file:
      sg = sg_data[qdata[qid]['imageId']]
    # Create each prompt
    if sg and len(sg)<=100:
      prompt = ["Instruction: Provide an answer to the question. Use the image and details to answer with ONE WORD ONLY.\n",
        qdata[qid]['imageId']+'.jpg',
        f"Details: {sg}",
        f"Question: {question} Answer:"
      ]
    else:
      count_no_sg += 1
      # prompt = ["Instruction: Provide an answer to the question. Use the image and details to answer with ONE WORD ONLY.\n",
      #   qdata[qid]['imageId']+'.jpg',
      #   f"Question: {question} Answer:"
      # ]
      prompt = ["Instruction: Provide an answer to the question. Use the image to answer.\n",
        qdata[qid]['imageId']+'.jpg',
        f"Question: {question} Answer:"
      ]

    if qdata[qid]['imageId'] not in q_data_img:
      q_data_img[qdata[qid]['imageId']] = []

    q_data_img[qdata[qid]['imageId']].append([prompt,qid])


  # Now change from image: [prompts] to batched prompts
  b = 0
  for img in q_data_img:
    prompts[b] = [x[0] for x in q_data_img[img]]
    qid_batched[b] = [x[1] for x in q_data_img[img]]
    b+=1

  if not os.path.exists(save_path):
    os.makedirs(save_path)

  with open(os.path.join(save_path, 'prompts_batched.json'), 'w+') as f:
    json.dump(prompts, f)

  with open(os.path.join(save_path, 'qid_batched.json'), 'w+') as f:
    json.dump(qid_batched, f)

  # with open(os.path.join(save_path, 'answers.json'), 'w+') as f:
  #   json.dump(answers, f)

  print('Number of batches:', len(prompts.keys()))

def batch_prompts(qfile, save_path, sg_file=None):
  prompts = {0:[]}
  qid_batched = {0:[]}
  batchsize = 4
  b_idx = 0

  with open(qfile,'r') as f:
    qdata = json.load(f)

  if sg_file:
    with open(sg_file, 'r') as f:
      sg_data = json.load(f)
    sg_size = []

  count_no_sg = 0

  for qid in qdata:
    # if batch is full, create new batch
    if len(prompts[b_idx])==batchsize:
      b_idx += 1
      prompts[b_idx] = []
      qid_batched[b_idx] = []
      
    question = qdata[qid]['question']
    sg = None

    if sg_file:
      if qdata[qid]['imageId']+'.jpg' not in sg_data:
        count_no_sg += 1
        sg = None
      else:
        sg = sg_data[qdata[qid]['imageId']+'.jpg']
    # Create each prompt
    # if sg and len(sg)<=100:
    if sg:
      prompt = ["Instruction: Provide an answer to the question. Use the image and details to answer with ONE WORD ONLY.\n",
        qdata[qid]['imageId']+'.jpg',
        f"Scene Details: {'. '.join(sg)}",
        f"Question: {question} Answer:"
      ]
      # Write prompt to batched prompts
      prompts[b_idx].append(prompt)
      qid_batched[b_idx].append(qid)
      # answers[b_idx].append(answer)
      sg_size.append(len(sg))
    elif not sg:
      # continue
      count_no_sg += 1
      # prompt = ["Instruction: Provide an answer to the question. Use the image and details to answer with ONE WORD ONLY.\n",
      #   qdata[qid]['imageId']+'.jpg',
      #   f"Question: {question} Answer:"
      # ]
      # prompt = ["Instruction: Provide an answer to the question. Use the image to answer with ONE WORD ONLY.\n",
      prompt = ["Instruction: Provide an answer to the question. Use the image to answer.\n",
        qdata[qid]['imageId']+'.jpg',
        f"Question: {question} Answer:"
      ]

      # Write prompt to batched prompts
      prompts[b_idx].append(prompt)
      qid_batched[b_idx].append(qid)
      # answers[b_idx].append(answer)

    # img.close()

  if not os.path.exists(save_path):
    os.makedirs(save_path)

  with open(os.path.join(save_path, 'prompts_batched.json'), 'w+') as f:
    json.dump(prompts, f)

  with open(os.path.join(save_path, 'qid_batched.json'), 'w+') as f:
    json.dump(qid_batched, f)

  # with open(os.path.join(save_path, 'answers.json'), 'w+') as f:
  #   json.dump(answers, f)

  print('Number of batches:', len(prompts.keys()))
  print('Number of no SG:', count_no_sg)

  if sg:
    print('Average SG size:', sum(sg_size)/len(sg_size), min(sg_size), max(sg_size))  # print(f'No SG for {count_no_sg}/20000 pairs')

# qpath = 'Dataset/GQA/questions/testdev_all_questions.json'
enum = 40
# qpath = 'Dataset/GQA/questions/val_sample_questions.json'
qpath = 'Dataset/GQA/questions/rel_only_bal/val_rel_questions.json'
# qpath = 'Dataset/GQA/questions/rel_only/val_rel_questions.json'

img2url = 'Dataset/GQA/img2url.json'
save_path = f'Experiments/Exp{enum}/'
# batch_prompts(qpath, img2url, save_path, sg_file='Dataset/GQA/sceneGraphs/val_sample_sg_tuples.json')
# batch_prompts(qpath, img2url, save_path, sg_file='Dataset/GQA/sceneGraphs/val_relbal_sg_tuples.json')
# batch_prompts(qpath, save_path, sg_file='Dataset/GQA/gen_sceneGraphs/val_relbal_sg_tuples.json')
batch_prompts(qpath, save_path)
# batch_prompts(qpath, img2url, save_path)
# batch_prompts_imgwise(qpath, save_path)
