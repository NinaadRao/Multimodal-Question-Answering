from datasets import Dataset, load_dataset
import json
from PIL import Image
import os

# Read questions and answers from JSON file
# with open('Dataset/GQA/questions/rel_only_bal/testdev_rel_questions.json','r') as f:
#     qdata = json.load(f)

# def load_and_close_images(image_paths):
#     images = []
#     for img_path in image_paths:
#         with Image.open(img_path) as img:
#             images.append(img.copy())
#     return images

# # Create a dictionary for Hugging Face dataset
# dataset_dict = {
#     "image": [],
#     "question": [],
#     "answer": [], 
# }

# datadir = 'Dataset/GQA/images/'
# for k in qdata.keys():
#     dataset_dict["image"].append(datadir+qdata[k]['imageId']+'.jpg')
#     dataset_dict["question"].append(qdata[k]['question'])
#     dataset_dict["answer"].append(qdata[k]['answer'])

# dataset_dict["image"] = load_and_close_images(dataset_dict["image"])

# # Convert dictionary to Hugging Face dataset
# dataset = Dataset.from_dict(dataset_dict)

# print(dataset)

# rel_bal_ds = load_dataset('imagefolder',data_dir='Dataset/GQA/HFDatadir')
# rel_bal_ds = load_dataset('csv',data_files={'train':'Dataset/GQA/HFDatadir/train_metadata.csv','test':'Dataset/GQA/HFDatadir/testdev_metadata.csv','validation':'Dataset/GQA/HFDatadir/val_metadata.csv'})
rel_bal_ds_sg = load_dataset('csv',data_files={'train':'Dataset/GQA/HFDatadir/SGFinetune/train_metadata.csv','validation':'Dataset/GQA/HFDatadir/SGFinetune/val_metadata.csv'})

print(rel_bal_ds_sg)

# print('Train:',len([x for x in os.listdir('Dataset/GQA/HFDatadir/train') if x.endswith('.jpg')]))
# print('Val:',len([x for x in os.listdir('Dataset/GQA/HFDatadir/val') if x.endswith('.jpg')]))
# print('Test:',len([x for x in os.listdir('Dataset/GQA/HFDatadir/test') if x.endswith('.jpg')]))

with open('AWS/huggingface.json','r') as f:
    hftoken = json.load(f)
    hftoken = hftoken['write']

# Push dataset to HuggingFace Hub
# from huggingface_hub import notebook_login
# notebook_login(token=hftoken)

# rel_bal_ds.pload_to_hf(repo_id="Relational_Balanced_GQA_Dataset", private=True)
# rel_bal_ds.push_to_hub("pragnyas/Balanced_Relational_GQA_full", private=True, token=hftoken)
rel_bal_ds_sg.push_to_hub("pragnyas/Balanced_Relational_GQA_SGFinetune", private=True, token=hftoken)

# print(test_ds['train'])