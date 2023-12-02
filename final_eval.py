import pandas as pd

df = pd.read_csv("router_test_balanced_gt_new2_01_123.csv")

preds = []
with open("test_labels_pred.txt") as f:
  for l in f:
    preds.append(int(l.strip()))

df['router_pred'] = preds
df['labels'] = classification_gt

temp_map = []
for i in range(len(df)):
  if df['router_pred'][i] == 0:
    temp_map.append(0)
  elif df['router_pred'][i] == 1:
    temp_map.append(3)
  elif df['router_pred'][i] == 2:
    temp_map.append(1)
  elif df['router_pred'][i] == 3:
    temp_map.append(2)

df['new_map'] = temp_map

reverse_mapping = {0: 't1_gt', 1: 't2_1gt', 2: 't2_2gt', 3: 't3_gt'}

final_pred = []
for i in range(len(df)):
  router_pred = df['new_map'][i]
  final_pred.append(df[reverse_mapping[router_pred]][i])

df['final_pred'] = final_pred

df.to_csv('final_capstone_rule3.2.csv', index=False)
