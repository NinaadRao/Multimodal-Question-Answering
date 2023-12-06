import os
import json
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import shutil
import random
import pandas as pd
from scipy import stats

# Organize Test Data
def organize_data(q_path, save_dir,sample=(-1,-1)):
    with open(q_path) as f:
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

    # Plot histograms for each categorical variable
    def plot_categorical_histogram(data, ax, title):
        unique, counts = np.unique(data, return_counts=True)
        ax.bar(unique, counts)
        ax.set_title(title)
        ax.set_xlabel('Categories')
        ax.set_ylabel('Count')
        ax.set_xticklabels(unique, rotation=45)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # plot_categorical_histogram(qtype['detailed'], axs[0], 'Detailed')
    plot_categorical_histogram(qtype['semantic'], axs[0], 'Semantic')
    plot_categorical_histogram(qtype['structural'], axs[1], 'Structural')

    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # plot_categorical_histogram(answers, axs[0], 'Answers')
    # plot_categorical_histogram(fullAnswer, axs[1], 'Full Answers')
    
    # print(len(set(answers)), len(answers))
    # print(len(set(fullAnswer)), len(fullAnswer))

    # with open('Dataset/GQA/testdev_answers_dist.txt', 'w') as f:
    #     f.write('\n'.join(list(set(answers))))
    plt.tight_layout()
    plt.show()

    return list(qdata.keys()),list(set(images))

def get_train_stats(qfolder):
    files = os.listdir(qfolder)

    questions = []
    images = []

    for f in files:
        q, i = get_split_stats(os.path.join(qfolder, f))
        questions.extend(q)
        images.extend(i)
        print(f"After file {f}: ", len(questions))

    print(f'Num questions: {len(set(questions))}')
    print(f'Num images: {len(set(images))}')


def get_sg_objects(sg_file, obj_map = dict()):
    with open(sg_file) as f:
        sg_data = json.load(f)

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

    return obj_map

def get_sg_sample(sg_file,split, img_folder):
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

    with open(f'Dataset/GQA/sceneGraphs/{split}_relbal_sceneGraphs_lt100.json', 'w+') as f:   
    # with open('Dataset/GQA/sceneGraphs/val_sample_sceneGraphs.json', 'w+') as f:   
        json.dump(new_sg, f) 

def get_sg_tuples(sg_file, split, verbalize=True):
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
                if verbalize:
                    details.append(f"{objVal['name']} is {attr}")
                else:
                    details.append([objVal['name'], attr])
            
            for rel in objVal['relations']:
                if verbalize:
                    details.append(f"{objVal['name']} {rel['name']} {obj_map[rel['object']]}")
                else:
                    details.append([objVal['name'], rel['name'], obj_map[rel['object']]])
        
        if len(details)>100:
            continue
        if verbalize:
            sg_tuples[k+'.jpg'] = '. '.join(details)
        else:
            sg_tuples[k+'.jpg'] = details
        sg_size.append(len(details))

    print('Scene Graph as Tuples Stats:')
    print(f'Average number of tuples: {np.mean(sg_size)}')
    print(f'Min number of tuples: {np.min(sg_size)}')
    print(f'Max number of tuples: {np.max(sg_size)}')
    print(f'Number of tuples: {len(sg_tuples)}')
    # with open('Dataset/GQA/sceneGraphs/val_sample_sg_tuples.json', 'w+') as f:
    with open(f'Dataset/GQA/sceneGraphs/{split}_relbal_sg_tuples_lt100.json', 'w+') as f:
    # with open('Dataset/GQA/sceneGraphs/val_relbal_sg_tuples_lt100.json', 'w+') as f:
        json.dump(sg_tuples, f)

    # print('Percentile of 100:', stats.percentileofscore(sg_size, 100))
    # plt.hist(sg_size, bins=100)
    # plt.show()

def get_sg_tuple_by_id(sg_tuples_file, id):
    with open(sg_tuples_file) as f:
        sg_tuples = json.load(f)
    
    print(sg_tuples[id])


def sample_balanced(q_path):
    q_data = None
    if os.path.isdir(q_path):
        files = os.listdir(q_path)
        
        for f in files:
            with open(os.path.join(q_path, f)) as f:
                q_data.update(json.load(f))
    else:
        with open(q_path) as f:
            q_data = json.load(f)

    print(len(q_data))

def sample_rel_only(q_path, save_path):
    q_data = dict()
    if os.path.isdir(q_path):
        files = os.listdir(q_path)
        
        for f in files:
            with open(os.path.join(q_path, f)) as f:
                q_data.update(json.load(f))
    else:
        with open(q_path) as f:
            q_data = json.load(f)

    new_q_data = dict()
    for k in q_data.keys():
        if q_data[k]['types']['semantic']=='rel':
            new_q_data[k] = q_data[k]

    filename = q_path.split('/')[-1]
    filename = filename.replace('.json','')
    filename = filename.replace('_all_','_rel_')
    filename += '.json'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path,filename), 'w+') as f:
        json.dump(new_q_data, f)
    
    print(f'For {q_path}: {len(new_q_data)}')


def create_dir_for_HF(qpath, datadir, split):
    with open(qpath) as f:
        qdata = json.load(f)

    if not os.path.exists(os.path.join(datadir,split)):
        os.makedirs(os.path.join(datadir,split))

    metadata = {'file_name':[],'question':[],'answer':[]}
    for k in qdata.keys():
        metadata['file_name'].append(qdata[k]['imageId']+'.jpg')
        metadata['question'].append(qdata[k]['question'])
        metadata['answer'].append(qdata[k]['answer'])

        shutil.copy(os.path.join('Dataset/GQA/images',qdata[k]['imageId']+'.jpg'), os.path.join(datadir,split))

    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(datadir,split,'metadata.csv'), index=False)

def create_csv_for_HFdata(qpath, datadir, split):
    with open(qpath) as f:
        qdata = json.load(f)

    if not os.path.exists(datadir):
        os.makedirs(datadir)

    metadata = {'image':[],'question':[],'answer':[]}
    for k in qdata.keys():
        metadata['image'].append(qdata[k]['imageId']+'.jpg')
        metadata['question'].append(qdata[k]['question'])
        metadata['answer'].append(qdata[k]['answer'])


    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(datadir,f'{split}_metadata.csv'), index=False)

def remove_images_withoutSG(metadata_file, sg_file):
    with open(sg_file) as f:
        sg_data = json.load(f)
        keys = list(sg_data.keys())

    metadata = pd.read_csv(metadata_file)
    # print(metadata['image'].iloc[0])
    print('len before:', len(metadata))
    metadata = metadata[metadata['image'].isin(keys)]
    print('len after:', len(metadata))

    save_path = metadata_file.split('/')[:-1]
    save_path.append('SGFinetune')
    save_path = '/'.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path += '/'+metadata_file.split('/')[-1]
    metadata.to_csv(save_path, index=False)


# qpath = 'Dataset/GQA/questions/testdev_all_questions.json'
# qpath = 'Dataset/GQA/questions/testdev_balanced_questions.json'
# qpath = 'Dataset/GQA/questions/val_all_questions.json'
# get_split_stats(qpath)
# organize_data(qpath, 'Dataset/GQA/split/testdev')
# organize_data(qpath, 'Dataset/GQA/split/val', sample=(10000,2))
# new_path = qpath.split('/')[:-1]
# new_path.append('val_sample_questions.json')
# new_path = '/'.join(new_path)
# get_split_stats(new_path)

# objMap = get_sg_objects('Dataset/GQA/sceneGraphs/val_sceneGraphs.json')
# objMap = get_sg_objects('Dataset/GQA/sceneGraphs/train_sceneGraphs.json', objMap)

# get_sg_sample('Dataset/GQA/sceneGraphs/val_sceneGraphs.json', 'val','Dataset/GQA/split/val_sample')
# get_sg_tuples('Dataset/GQA/sceneGraphs/val_sample_sceneGraphs.json','val', verbalize=True)
# get_sg_tuple_by_id('Dataset/GQA/sceneGraphs/val_sample_sg_tuples.json', '2363338')

get_sg_sample('Dataset/GQA/sceneGraphs/val_sceneGraphs.json', 'val','Dataset/GQA/split/val_rel_bal')
get_sg_tuples('Dataset/GQA/sceneGraphs/val_relbal_sceneGraphs_lt100.json', 'val', verbalize=True)
# get_sg_sample('Dataset/GQA/sceneGraphs/train_sceneGraphs.json', 'train','Dataset/GQA/HFDatadir/train')
# get_sg_tuples('Dataset/GQA/sceneGraphs/train_relbal_sceneGraphs_lt100.json', 'train', verbalize=True)

# Combine val and train sg_tuples
# train_sg = 'Dataset/GQA/sceneGraphs/train_relbal_sg_tuples_lt100.json'
# val_sg = 'Dataset/GQA/sceneGraphs/val_relbal_sg_tuples_lt100.json'

# with open(train_sg) as f:
#     train_sg_data = json.load(f)

# with open(val_sg) as f:
#     val_sg_data = json.load(f)

# train_sg_data.update(val_sg_data)

# with open('Dataset/GQA/sceneGraphs/all_relbal_sg_tuples_lt100.json', 'w+') as f:
#     json.dump(train_sg_data, f)

# remove_images_withoutSG('Dataset/GQA/HFDatadir/train_metadata.csv', 'Dataset/GQA/sceneGraphs/all_relbal_sg_tuples_lt100.json')
# remove_images_withoutSG('Dataset/GQA/HFDatadir/val_metadata.csv', 'Dataset/GQA/sceneGraphs/all_relbal_sg_tuples_lt100.json')

# get_train_stats('Dataset/GQA/questions/train_all_questions')

# print('Testdev:')
# get_split_stats('Dataset/GQA/questions/testdev_balanced_questions.json')
# print('\n\n')

# print('Val:')
# get_split_stats('Dataset/GQA/questions/val_balanced_questions.json')
# print('\n\n')

# print('Train:')
# get_split_stats('Dataset/GQA/questions/train_balanced_questions.json')
# print('\n\n')


# sample_balanced('Dataset/GQA/questions/testdev_balanced_questions.json')
# sample_balanced('Dataset/GQA/questions/train_all_questions')

# sample_rel_only('Dataset/GQA/questions/val_all_questions.json', 'Dataset/GQA/questions/rel_only')
# sample_rel_only('Dataset/GQA/questions/testdev_all_questions.json', 'Dataset/GQA/questions/rel_only')
# sample_rel_only('Dataset/GQA/questions/train_all_questions', 'Dataset/GQA/questions/rel_only')

# sample_rel_only('Dataset/GQA/questions/val_balanced_questions.json', 'Dataset/GQA/questions/rel_only_bal')
# sample_rel_only('Dataset/GQA/questions/testdev_balanced_questions.json', 'Dataset/GQA/questions/rel_only_bal')
# sample_rel_only('Dataset/GQA/questions/train_balanced_questions.json', 'Dataset/GQA/questions/rel_only_bal')

# print('Testdev:')
# get_split_stats('Dataset/GQA/questions/rel_only_bal/testdev_rel_questions.json')
# print('\n\n')

# print('Val:')
# get_split_stats('Dataset/GQA/questions/rel_only_bal/val_rel_questions.json')
# print('\n\n')

# print('Train:')
# get_split_stats('Dataset/GQA/questions/rel_only_bal/train_rel_questions.json')
# print('\n\n')

# organize_data('Dataset/GQA/questions/rel_only_bal/val_rel_questions.json', 'Dataset/GQA/split/val_rel_bal')

# create_dir_for_HF(qpath='Dataset/GQA/questions/rel_only_bal/testdev_rel_questions.json', 
#                   datadir='Dataset/GQA/HFDatadir', 
#                   split='testdev')

# create_dir_for_HF(qpath='Dataset/GQA/questions/rel_only_bal/train_rel_questions.json', 
#                   datadir='Dataset/GQA/HFDatadir', 
#                   split='train')

# create_dir_for_HF(qpath='Dataset/GQA/questions/rel_only_bal/val_rel_questions.json', 
#                   datadir='Dataset/GQA/HFDatadir', 
#                   split='val')

# create_csv_for_HFdata(qpath='Dataset/GQA/questions/rel_only_bal/testdev_rel_questions.json', 
#                   datadir='Dataset/GQA/HFDatadir', 
#                   split='testdev')

# create_csv_for_HFdata(qpath='Dataset/GQA/questions/rel_only_bal/train_rel_questions.json', 
#                   datadir='Dataset/GQA/HFDatadir', 
#                   split='train')

# create_csv_for_HFdata(qpath='Dataset/GQA/questions/rel_only_bal/val_rel_questions.json', 
#                   datadir='Dataset/GQA/HFDatadir', 
#                   split='val')