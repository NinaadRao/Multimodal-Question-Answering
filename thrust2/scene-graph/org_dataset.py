import os
import json
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import shutil
import random

# Organize Test Data
def organize_data(q_path, save_dir,sample=(-1,-1)):
    with open(qpath) as f:
        qdata = json.load(f)
    

    # TASKS:
    images = []
    for k in qdata.keys():
        # print(k, qdata[k])
        # break

        # Get all image IDs (imageId)
        images.append(qdata[k]['imageId'])
        # Get question ID (key)
        # Get question (question)
        # Get type (types) storing detailed, semantic and structural
        # Aux: Get all types in dataset

    img_dir = 'Dataset/GQA/images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images = list(set(images))
    # Sample sample[0] images if sample[0] is not -1
    if sample[0]!=-1:
        np.random.seed(42)
        images = np.random.choice(images, sample[0], replace=False)
        print('Sampled images:', len(images))

    # Sample sample[1] questions per image if sample[1] is not -1
    if sample[1]!=-1:
        img_question_map = {i:sample[1] for i in images}
        new_json = dict()
        qkeys = list(qdata.keys())
        random.seed(42)
        random.shuffle(qkeys)
        
        for k in qkeys:
            if qdata[k]['imageId'] in img_question_map:
                if img_question_map[qdata[k]['imageId']]>0:
                    new_json[k] = qdata[k]
                    img_question_map[qdata[k]['imageId']] -= 1
            else:
                continue
        
        new_path = q_path.split('/')[:-1]
        new_path.append('val_sample_questions.json')
        with open('/'.join(new_path), 'w+') as f:
            json.dump(new_json, f)
        print('Sampled Questions: ', len(new_json))
        

    for i in images:
        img_path = os.path.join(img_dir, i+'.jpg')
        shutil.copy(img_path, save_dir)

def get_split_stats(qpath, only_len=False):
    with open(qpath) as f:
        qdata = json.load(f)
    
    if only_len:
        print('Total number of questions:',len(qdata))
        return

    # TASKS:
    images = []
    qtype = {'detailed':[],'semantic':[],'structural':[]}
    answers = []
    fullAnswer = []
    for k in qdata.keys():
        # Get all image IDs (imageId)
        images.append(qdata[k]['imageId'])

        # Get types:
        qtype['detailed'].append(qdata[k]['types']['detailed'])
        qtype['semantic'].append(qdata[k]['types']['semantic'])
        qtype['structural'].append(qdata[k]['types']['structural'])

        answers.append(qdata[k]['answer'])
        fullAnswer.append(qdata[k]['fullAnswer'])

    # Get split stats:
    print('Total questions: ', len(qdata.keys()))
    print('Total unique images: ', len(set(images)))
    print('Total unique detailed types: ', len(set(qtype['detailed'])))
    print('Total unique semantic types: ', len(set(qtype['semantic'])))
    print('Total unique structural types: ', len(set(qtype['structural'])))

    # Visualize distribution of qtypes:
    # Plot histogram of qtypes:
    # Create a figure with three subplots in a row
    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot histograms for each categorical variable
    def plot_categorical_histogram(data, ax, title):
        unique, counts = np.unique(data, return_counts=True)
        ax.bar(unique, counts)
        ax.set_title(title)
        ax.set_xlabel('Categories')
        ax.set_ylabel('Count')
        ax.set_xticklabels(unique, rotation=45)

    # plot_categorical_histogram(qtype['detailed'], axs[0], 'Detailed')
    # plot_categorical_histogram(qtype['semantic'], axs[0], 'Semantic')
    # plot_categorical_histogram(qtype['structural'], axs[1], 'Structural')

    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # plot_categorical_histogram(answers, axs[0], 'Answers')
    # plot_categorical_histogram(fullAnswer, axs[1], 'Full Answers')
    
    print(len(set(answers)), len(answers))
    print(len(set(fullAnswer)), len(fullAnswer))

    with open('Dataset/GQA/testdev_answers_dist.txt', 'w') as f:
        f.write('\n'.join(list(set(answers))))
    # plt.tight_layout()
    # plt.show()

def get_sg_objects(sg_file):
    with open(sg_file) as f:
        sg_data = json.load(f)

    obj_map = dict()
    for k in sg_data.keys():
        # print(k, sg_data[k])
        for objKey,objVal in sg_data[k]['objects'].items():
            # Object key and name is unique
            # if objKey in obj_map and obj_map[objKey]!=objVal['name']:
            #     print(objKey, obj_map[objKey], objVal['name'])
            obj_map[objKey] = objVal['name']
    
    print(f'Found {len(obj_map)} objects')
    with open('Dataset/GQA/sg_obj_map.json', 'w+') as f:
        json.dump(obj_map, f)

def get_sg_sample(sg_file,img_folder):
    img_names = os.listdir(img_folder)
    img_names = [x for x in img_names if x.endswith('.jpg')]

    with open(sg_file) as f:
        sg_data = json.load(f)

    new_sg = dict()
    for k in sg_data.keys():
        if k+'.jpg' in img_names:
            new_sg[k] = sg_data[k]

    print(f'Num images: {len(img_names)}')
    print(f'Num scene graphs: {len(new_sg)}')

    with open('Dataset/GQA/sceneGraphs/val_sample_sceneGraphs.json', 'w+') as f:   
        json.dump(new_sg, f) 

def get_sg_tuples(sg_file):
    with open(sg_file) as f:
        sg_data = json.load(f)

    if not os.path.exists('Dataset/GQA/sg_obj_map.json'):
        print('Object map not found. Create one using original SG file.')
        return
    
    with open('Dataset/GQA/sg_obj_map.json') as f:
        obj_map = json.load(f)

    sg_tuples = dict()
    sg_size = []
    for k in sg_data.keys():
        details = []
        for objKey,objVal in sg_data[k]['objects'].items():
            for attr in objVal['attributes']:
                details.append([objVal['name'], attr])
            
            for rel in objVal['relations']:
                details.append([objVal['name'], rel['name'], obj_map[rel['object']]])
        
        sg_tuples[k] = details
        sg_size.append(len(details))

    print('Scene Graph as Tuples Stats:')
    print(f'Average number of tuples: {np.mean(sg_size)}')
    print(f'Min number of tuples: {np.min(sg_size)}')
    print(f'Max number of tuples: {np.max(sg_size)}')
    with open('Dataset/GQA/sceneGraphs/val_sample_sg_tuples.json', 'w+') as f:
        json.dump(sg_tuples, f)

def get_sg_tuple_by_id(sg_tuples_file, id):
    with open(sg_tuples_file) as f:
        sg_tuples = json.load(f)
    
    print(sg_tuples[id])
# qpath = 'Dataset/GQA/questions/testdev_all_questions.json'
# qpath = 'Dataset/GQA/questions/val_all_questions.json'
# qpath = 'Dataset/GQA/questions/testdev_balanced_questions.json'
# qpath = 'Dataset/GQA/questions/val_all_questions.json'
# get_split_stats(qpath)
# organize_data(qpath, 'Dataset/GQA/split/testdev')
# organize_data(qpath, 'Dataset/GQA/split/val', sample=(10000,2))
# new_path = qpath.split('/')[:-1]
# new_path.append('val_sample_questions.json')
# new_path = '/'.join(new_path)
# get_split_stats(new_path)

# get_sg_objects('Dataset/GQA/sceneGraphs/val_sceneGraphs.json')
# get_sg_sample('Dataset/GQA/sceneGraphs/val_sceneGraphs.json', 'Dataset/GQA/split/val_sample')
# get_sg_tuples('Dataset/GQA/sceneGraphs/val_sample_sceneGraphs.json')
get_sg_tuple_by_id('Dataset/GQA/sceneGraphs/val_sample_sg_tuples.json', '21')