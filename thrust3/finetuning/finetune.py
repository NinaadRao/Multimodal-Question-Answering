import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
import torchvision.transforms as transforms
import json
import random
import wandb
import os

def load_model(checkpoint):
    # Here we skip some special modules that can't be quantized properly
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["lm_head", "embed_tokens"],
    )

    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint, use_auth_token=False)

    # print(model)

    return model, processor


def check_inference(model, processor, prompts, max_new_tokens=50):
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    if len(bad_words) > 0:
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids

    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, eos_token_id=[eos_token_id], bad_words_ids=bad_words_ids, max_new_tokens=max_new_tokens, early_stopping=True)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


# return a list of answers to a question
def process_answers(answer_list):
    res_answer = []
    for p in answer_list:
        res_answer.append(p['answer'])
    return res_answer


def process_question_file(question_path):
    #format: question_id: (image_id, question)
    question_dict = {}
    # process question file
    # Open the JSON file in read mode
    with open(question_path, 'r') as json_file:
        # Read the JSON data into a Python dictionary or list
        question_file = json.load(json_file)
    questions = question_file['questions']
    for p in questions:
        image_id = p['image_id']
        question_id = p['question_id']
        question = p['question']
        question_dict[question_id] = (image_id, question)
    return question_dict


def process_annotation_file(annotation_path, question_dict):
    # format: question_id: (image_id, question, answer_list)
    annotation_dict = {}
    # final output: image, question, answer
    # process annotation file
    # Open the JSON file in read mode
    with open(annotation_path, 'r') as json_file:
        # Read the JSON data into a Python dictionary or list
        anno_file = json.load(json_file)

    annotations = anno_file['annotations']
    for e in annotations:
        image_id = e['image_id']
        question_id = e['question_id']
        question = question_dict[question_id][1]
        answers = e['answers']
        answer_list = process_answers(answers)
        annotation_dict[question_id] = (image_id, question, answer_list)
    return annotation_dict


def process_annotation_file_dataset(annotation_path, question_dict):
    data_samples = {"image_ids": [], "questions": [], "answer_lists": []}
    # final output: dict of image, question, answer lists
    # process annotation file
    # Open the JSON file in read mode
    with open(annotation_path, 'r', encoding='utf-8') as json_file:
        # Read the JSON data into a Python dictionary or list
        anno_file = json.load(json_file)

    annotations = anno_file['annotations']
    for e in annotations:
        image_id = e['image_id']
        question_id = e['question_id']
        question = question_dict[question_id][1]
        answers = e['answers']
        answer_list = process_answers(answers)
        data_samples["image_ids"].append(image_id)
        data_samples["questions"].append(question)
        data_samples["answer_lists"].append(answer_list)
    return data_samples
    

def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

def ds_transforms(example_batch):
    image_size = processor.image_processor.image_size
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    image_transform = transforms.Compose([
        convert_to_rgb,
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    prompts = []
    for i in range(len(example_batch['image_ids'])):

        question = example_batch['questions'][i]
        image_id = example_batch['image_ids'][i]
        padded_image_id = f"{image_id:012d}"
        answer_list = example_batch['answer_lists'][i]

        # sample: https://okvqa11632.s3.amazonaws.com/val2014/COCO_val2014_000000000042.jpg
  
        # use the first answer temporarily
        prompts.append(
            [
                "Instruction: provide a short answer to the question. Use the image to answer.\n",
                "https://okvqa11632.s3.amazonaws.com/train2014/COCO_train2014_"+padded_image_id+".jpg",
                # "Question: {} Answer: {}.\n".format(question, answer_list[0])
                "Question: {} Answer: {}.\n".format(question, random.choice(answer_list))
            ],
        )

    inputs = processor(prompts, transform=image_transform, return_tensors="pt").to(device)

    inputs["labels"] = inputs["input_ids"]

    return inputs


if __name__ == "__main__":

    print("--------------   import done   ---------------")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = "HuggingFaceM4/idefics-9b"
    # checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    # checkpoint = "HuggingFaceM4/tiny-random-idefics"
    model, processor = load_model(checkpoint)

    print("--------------   model loaded   ---------------")

    # check inference
    url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
    prompts = [
        "Instruction: provide an answer to the question. Use the image to answer.\n",
        url,
        "Question: What's on the picture? Answer:",
    ]
    check_inference(model, processor, prompts, max_new_tokens=5)

    # preprocess
    # need to change to training set
    question_path =  'data/OpenEnded_mscoco_train2014_questions.json'
    annotation_path =  'data/mscoco_train2014_annotations.json'

    question_dict = process_question_file(question_path)
    data_samples = process_annotation_file_dataset(annotation_path, question_dict)

    print("--------------   data loaded   ---------------")

    # dataset
    ds = Dataset.from_dict(data_samples)
    ds = ds.train_test_split(test_size=0.2)
    train_ds = ds["train"].shuffle(seed=42)
    eval_ds = ds["test"]
    train_ds.set_transform(ds_transforms)
    eval_ds.set_transform(ds_transforms)
    print("--------------   dataset done   ---------------")
    # import pdb; pdb.set_trace()
    

    # LoRA
    model_name = checkpoint.split("/")[1]
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.load_adapter
    model.print_trainable_parameters()

    # training
    # training_args = TrainingArguments(
    #     output_dir="output",
    #     learning_rate=2e-4,
    #     bf16=True,
    #     per_device_train_batch_size=2,
    #     per_device_eval_batch_size=2,
    #     gradient_accumulation_steps=8,
    #     dataloader_pin_memory=False,
    #     save_total_limit=3,
    #     evaluation_strategy="steps",
    #     save_strategy="steps",
    #     save_steps=300,
    #     eval_steps=300,
    #     logging_steps=100,
    #     max_steps=3600,
    #     remove_unused_columns=False,
    #     push_to_hub=True,
    #     label_names=["labels"],
    #     load_best_model_at_end=True,
    #     report_to=None,
    #     optim="paged_adamw_8bit",
    # )
    os.environ["WANDB_PROJECT"] = "capstone-okvqa" # log to your project 
    os.environ["WANDB_LOG_MODEL"] = "all" # log your models

    training_args = TrainingArguments(
        output_dir=f"{model_name}-okvqa-v2",
        run_name=f"{model_name}-okvqa-v2",
        learning_rate=5e-4,
        bf16=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=16,
        dataloader_pin_memory=False,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=50,
        eval_steps=50,
        logging_steps=10,
        max_steps=1000,
        remove_unused_columns=False,
        push_to_hub=True,
        label_names=["labels"],
        load_best_model_at_end=True,
        report_to="wandb",
        optim="paged_adamw_8bit",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()

    model.push_to_hub(f"{model_name}-okvqa-v2", private=False)
    import pdb; pdb.set_trace()
