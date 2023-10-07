import json
import os
from PIL import Image

def batch_prompts(qfile, img2url, save_path):
  prompts = {0:[]}
  qid_batched = {0:[]}
  # answers = {0:[]}
  batchsize = 32
  b_idx = 0
  imdir = 'Dataset/GQA/split/val'

  with open(qfile,'r') as f:
    qdata = json.load(f)

  with open(img2url, 'r') as f:
    img2url = json.load(f)

  for qid in qdata:
    # if batch is full, create new batch
    if len(prompts[b_idx])==batchsize:
      b_idx += 1
      prompts[b_idx] = []
      qid_batched[b_idx] = []
      # answers[b_idx] = []

    # Get Google Drive URL of the image and question
    # imgurl = img2url[qdata[qid]['imageId']]
    # impath = os.path.join(imdir, qdata[qid]['imageId']+'.jpg')
    # img = Image.open(impath)
    question = qdata[qid]['question']
    answer = qdata[qid]['answer']

    # Create each prompt
    prompt = ["Instruction: Provide ONLY A ONE WORD ANSWER to the question. Use the image to answer with ONE WORD ONLY.\n",
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

# qpath = 'Dataset/GQA/questions/testdev_all_questions.json'
qpath = 'Dataset/GQA/questions/val_sample_questions.json'
img2url = 'Dataset/GQA/img2url.json'
save_path = 'Experiments/Exp3/'
batch_prompts(qpath, img2url, save_path)