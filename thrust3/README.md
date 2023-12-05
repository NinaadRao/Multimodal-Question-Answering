## MQA Thrust 3: okvqa
This project aims to explore IDEFICS large language model as the implicit knowledge base for knowledge-based VQA

## Prerequisites

Install required packages by running

`pip install -r requirements.txt`

## Prompting

n is the number of examples we want to include in the prompt, k is the number of answer candidates for each example

For pica prompting:
* prompts are formated with the PICa template
`python prompting/generate_prompt.py pica n`

For answer-aware based prompting:
* few shot examples are extracted by MCAN VQA model based on question+image embedding similarity
`python prompting/generate_prompt.py answer_aware n`

For answer-candidates based prompting:
* answer candidates and confidence scores are added as part of the prompts
`python prompting/generate_prompt.py answer_candidates n k`

For question-aware based prompting:
* few shot examples are extracted by BERT based on question embedding similarities
`python prompting/generate_prompt.py question_aware n`

For IDEFICS embedding based prompting:
* First generate the embedding for the examples in the retrieval pool using `prompting/generate_embed.py`
* Then get the inference results using `prompting/inference_sample_select.py`

## Finetune
To finetune the base model using the OKVQA training set with LoRA, use script `finetuning/finetune.py`.
The training hyperparameters can be set in the script. Around 20GB GPU memory is needed to finetune the idefics-9b model locally.

