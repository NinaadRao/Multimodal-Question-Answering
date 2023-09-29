import json
import os
import re
import nltk
import pandas as pd


def sanity_check(raw_outputs, prompts):
    # Check is answers and questions align
    with open(raw_outputs,'r') as f:
        preds = json.load(f)

    with open(prompts,'r') as f:
        prompts = json.load(f)

    if len(preds)!=len(prompts):
        print('Number of batches do not match')
        return False
    
    for i in range(len(preds)):
        i = str(i)
        if len(preds[i])!=len(prompts[i]):
            print(f'Number of elements in batch {i} do not match')
            return False
        
        for j in range(len(preds[i])):
            question = prompts[i][j][2]
            questions = re.findall(r'Question: (.*) Answer:', question)
            if questions[0] not in preds[i][j]:
                print(f'Question of {j}th case in batch {i} does not match')
                return False
            
    print('All good! Your answers and questions align.')
    return True


def parse_answer(raw_outputs):
    with open(raw_outputs,'r') as f:
        preds = json.load(f)

    answers = dict()
    for k in preds.keys():
        k = str(k)
        answers[k] = []

        for i in range(len(preds[k])):
            question = preds[k][i]

            # Fine string between 'Answer: ' and '\n'
            answer = re.findall(r'Answer: (.*)\n', question)
            ans = ''
            max_len = 0
            for a in answer:
                if len(a)>max_len:
                    ans = a
                    max_len = len(a)

            # Clean answer:
            ans = ans.lower()

            # Remove punctuation:
            ans = re.sub(r'[^\w\s]','',ans)

            # Remove stopwords using nltk:
            stop_words = set(nltk.corpus.stopwords.words('english'))
            word_tokens = nltk.word_tokenize(ans)
            ans = [w for w in word_tokens if not w in stop_words]
            ans = ' '.join(ans)
            answers[k].append(ans)

    with open('Experiments/Exp2/answers.json', 'w+') as f:
        json.dump(answers, f)
    
    return answers

def compute_accuracy(answers, qid, only_balanced=False, create_file=None):
    with open(answers,'r') as f:
        pred = json.load(f)

    with open(qid,'r') as f:    
        qid = json.load(f)

    with open('Dataset/GQA/questions/testdev_all_questions.json','r') as f:
        qdata = json.load(f)

    count_correct = 0
    perf_match = 0
    count_all = 0
    all_data = dict()

    for b in pred:
        all_data[b] = []

        for i in range(len(pred[b])):
            qid_val = qid[b][i]
            pred_val = pred[b][i]
            ans = qdata[str(qid_val)]['answer'].lower()
            balanced = qdata[str(qid_val)]['isBalanced']
            question = qdata[str(qid_val)]['question']

            if ans in pred_val:
                if only_balanced:
                    if balanced:
                        count_correct += 1
                        if ans==pred_val:
                            perf_match += 1
                else:
                    count_correct += 1
                    if ans==pred_val:
                        perf_match += 1

            count_all += 1

            # Store question, answer, if part of balanced set, and if IDEFICs answer was perfect match
            all_data[b].append((question, ans, balanced, int(ans==pred_val)))

    print(f'Accuracy: {count_correct/count_all} ({count_correct}/{count_all})') 
    print(f'Perfect Match Accuracy: {perf_match/count_all} ({perf_match}/{count_all})')

    if create_file:
        # Create a CSV comparing pred and gt_answer
        df = {'gt_answer':[], 'pred':[], 'question':[], 'balanced':[], 'perfect_match':[]}

        for b in all_data:
            for i in range(len(all_data[b])):
                df['question'].append(all_data[b][i][0])
                df['gt_answer'].append(all_data[b][i][1])
                df['pred'].append(pred[b][i])  
                df['balanced'].append(all_data[b][i][2])
                df['perfect_match'].append(all_data[b][i][3])

        df = pd.DataFrame(df)
        df.to_csv(os.path.join(create_file,'err_analysis.csv'), index=False)

if __name__=='__main__':
    assert sanity_check('Experiments/Exp2/raw_outputs.json', 'Experiments/Exp2/prompts_batched.json')

    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # answers = parse_answer('Experiments/Exp2/raw_outputs.json')

    compute_accuracy('Experiments/Exp2/answers.json', 'Experiments/Exp2/qid_batched.json', create_file='Experiments/Exp2/')