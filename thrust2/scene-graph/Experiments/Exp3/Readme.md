# Experiment 3
Data: val_sample_questions.json
Sampled images and question using random seed 43
10k images, 20k questions
model: IDEFICs
Batchsize: 64
313 Batches total

Prompt:
prompt = ["Instruction: Provide ONLY A ONE WORD ANSWER to the question. Use the image to answer with ONE WORD ONLY.\n",
      qdata[qid]['imageId']+'.jpg',
      f"Question: {question} Answer:"
    ]