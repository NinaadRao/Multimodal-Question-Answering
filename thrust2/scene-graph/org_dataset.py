import os
import json
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import shutil

# Organize Test Data
def organize_data(q_path, save_dir):
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

    for i in set(images):
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

qpath = 'Dataset/GQA/questions/testdev_all_questions.json'
# qpath = 'Dataset/GQA/questions/testdev_balanced_questions.json'
# qpath = 'Dataset/GQA/questions/val_all_questions.json'
get_split_stats(qpath)
# organize_data(qpath, 'Dataset/GQA/split/testdev')