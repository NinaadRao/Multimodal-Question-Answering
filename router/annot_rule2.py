import pandas as pd
import json

df = pd.read_csv("router_test_balanced_gt_new2_01_123.csv")

mapping = {'clevr': 0, 'vqacpv2': 0, 'tallyqa': 2, 'gqa': 1, 'okvqa': 3, 'aokvqa': 3}

rankings = {'clevr': [0,1,3,2], 'vqacpv2': [1,3,0,2], 'gqa': [1,3,0,2], 'tallyqa': [2,3,0,1], 'okvqa': [3,1,0,2], 'aokvqa': [1,3,2,0]}
reverse_mapping = {0: 't1_gt', 1: 't2_1gt', 2: 't2_2gt', 3: 't3_gt'}

image_path = []
classification_gt = []


for i in range(len(df)):

  if df['t1_gt'][i] == 0 and df['t2_1gt'][i] == 0 and df['t2_2gt'][i]  == 0 and df['t3_gt'][i] == 0:
    if df['sample_dataset'][i] == 'clevr' or df['sample_dataset'][i] == 'vqacpv2':
      classification_gt.append([0])
    elif df['sample_dataset'][i] == 'tallyqa':
      classification_gt.append([2])
    elif df['sample_dataset'][i] == 'gqa':
      classification_gt.append([1])
    else:
      classification_gt.append([3])

  else:
    temp = []
    if df['t1_gt'][i] == 1:
      temp.append(0)
    if df['t2_1gt'][i] == 1:
      temp.append(1)
    if df['t2_2gt'][i] == 1:
      temp.append(2)
    if df['t3_gt'][i] == 1:
      temp.append(3)
    # classification_gt.append([int(df['t1_gt'][i]), int(df['t2_1gt'][i]), int(df['t2_2gt'][i]), int(df['t3_gt'][i])])
    classification_gt.append(temp)

df['label'] = classification_gt


datasets = ['clevr', 'vqacpv2', 'tallyqa', 'gqa', 'okvqa']

#for splitting into train and val
test_df = pd.DataFrame()
for dataset in datasets:
    samples = df[df['sample_dataset'] == dataset].sample(1000)
    test_df = pd.concat([test_df, samples])
    df = df.drop(samples.index)

# saving
with open('router_train_final_rule3.jsonl', 'w') as file:
    for _, row in df.iterrows():
        json.dump(row.to_dict(), file)
        file.write('\n')

