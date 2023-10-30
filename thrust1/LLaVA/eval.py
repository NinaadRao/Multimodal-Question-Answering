import pandas as pd

df = pd.read_csv('one-word-final.csv')
def evaluate_result(predicted_answer, ground_truth_answer):
    dict_map = {'one' : '1',
                'two': '2',
                'three': '3',
                'four': '4',
                'five': '5',
                'six': '6',
                'seven': '7',
                'eight': '8',
                'nine': '9'
               }
    predicted_answer = predicted_answer.lower()
    splitted = predicted_answer.split()

    for i in range(len(splitted)):
        if splitted[i] in dict_map:
            splitted[i] = dict_map[splitted[i]]
    predicted_answer = ' '.join(splitted)
    if ground_truth_answer.lower() in predicted_answer.lower():
        return True
    else:
        return False


df['correct_new'] = df.apply(lambda x: evaluate_result(x.predicted_ans, x.ground_truth_ans), axis=1)




print(df['correct_new'].sum()/len(df['correct_new']))
pd.DataFrame(df).to_csv('outputs-eval-new.csv', index = False)


