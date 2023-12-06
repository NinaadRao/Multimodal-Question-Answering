import json
import os
import re
import nltk
import pandas as pd
import argparse
import random


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


def parse_answer(raw_outputs,save_answers_path):
    with open(raw_outputs,'r') as f:
        preds = json.load(f)

    answers = dict()
    for k in preds.keys():
        k = str(k)
        answers[k] = []

        for i in range(len(preds[k])):
            question = preds[k][i]

            # Fine string between 'Answer: ' and '\n'
            answer = re.findall(r'Answer:(.*?)(?:\.|Question:|\n|$)',question)
            if not answer:
                ans = ''
            else:
                ans = answer[0]

            # Clean answer:
            ans = ans.lower()

            # Remove punctuation:
            ans = re.sub(r'[^\w\s]','',ans)

            # Remove stopwords using nltk:
            # stop_words = set(nltk.corpus.stopwords.words('english'))
            # word_tokens = nltk.word_tokenize(ans)
            # ans = [w for w in word_tokens if not w in stop_words]
            
            # # Remove plural form
            # lemmatizer = nltk.stem.WordNetLemmatizer()
            # ans = [lemmatizer.lemmatize(word, pos='n') for word in ans]
            
            # ans = ' '.join(ans)
            ans = ans.strip()
            answers[k].append(ans)

    with open(save_answers_path, 'w+') as f:
        json.dump(answers, f)
    
    return answers

def compute_accuracy(answers, qid, only_balanced=False, create_file=None, batchlim=float('inf')):
    with open(answers,'r') as f:
        pred = json.load(f)

    with open(qid,'r') as f:    
        qid = json.load(f)

    # with open('Dataset/GQA/questions/rel_only/testdev_rel_questions.json','r') as f:
    with open('Dataset/GQA/questions/rel_only_bal/val_rel_questions.json','r') as f:
    # with open('Dataset/GQA/questions/rel_only_bal/testdev_rel_questions.json','r') as f:
    # with open('Dataset/GQA/questions/val_sample_questions.json','r') as f:
        qdata = json.load(f)

    count_correct = 0
    perf_match = 0
    count_all = 0
    all_data = dict()

    with open('Experiments/Exp33/count_acc_RelTR_50p.json') as f:
        acc_goodSG = json.load(f)

    with open('count_acc_RelTR_50p_qid.json','r') as f:
        questions = set(json.load(f)['qid'])

    # questions = {'qid':[]}
    for b in pred:
        if int(b)>=batchlim:
            break
        # print(len(pred[b]))
        # break
        all_data[b] = []

        for i in range(len(pred[b])):
            # if acc_goodSG[b][i]==False:
            #     continue
            if qid[b][i] not in questions:
            # if qid[b][i] in questions:
                continue
            
            # questions['qid'].append(qid[b][i])
            qid_val = qid[b][i]
            pred_val = pred[b][i]
            ans = qdata[str(qid_val)]['answer'].lower()
            balanced = qdata[str(qid_val)]['isBalanced']
            question = qdata[str(qid_val)]['question']

            if random.random()<0.001:
                with open('Experiments/Exp33/prompts_batched.json','r') as f:
                    prompts = json.load(f)
                print(prompts[b][i])
                print('Predicted: ',pred[b][i])
                print('GT: ',ans)
                print('\n\n\n')

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

    # with open('count_acc_RelTR_50p_qid.json','w+') as f:
    #     json.dump(questions,f)
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

def compute_accuracy_by_answer_type(answers, qid, only_balanced=False, create_file=None, batchlim=float('inf')):
    with open(answers,'r') as f:
        pred = json.load(f)

    with open(qid,'r') as f:    
        qid = json.load(f)

    with open('Dataset/GQA/questions/rel_only_bal/val_rel_questions.json','r') as f:
    # with open('Dataset/GQA/questions/rel_only_bal/testdev_rel_questions.json','r') as f:
    # with open('Dataset/GQA/questions/val_sample_questions.json','r') as f:
        qdata = json.load(f)

    count_correct = {'bin':0,'open':0}
    perf_match = {'bin':0,'open':0}
    count_all = {'bin':0,'open':0}
    all_data = dict()

    with open('Experiments/Exp33/count_acc_RelTR_50p.json') as f:
        acc_goodSG = json.load(f)

    for b in pred:
        if int(b)>=batchlim:
            break
        # print(len(pred[b]))
        # break
        all_data[b] = []
        for i in range(len(pred[b])):
            if acc_goodSG[b][i]==False:
                continue
            # print(f'keep {b},{i}')
            qid_val = qid[b][i]
            pred_val = pred[b][i]
            ans = qdata[str(qid_val)]['answer'].lower()
            balanced = qdata[str(qid_val)]['isBalanced']
            question = qdata[str(qid_val)]['question']

            if ans=='':
                continue

            key = 'bin' if  ans in ['yes','no'] else 'open'
            if 'or' in question:
                key = 'bin'
            correct = False
            if ans in pred_val:
                if only_balanced:
                    if balanced:
                        count_correct[key] += 1
                        correct = True
                        if ans==pred_val:
                            perf_match[key] += 1
                else:
                    count_correct[key] += 1
                    correct = True
                    if ans==pred_val:
                        perf_match[key] += 1

            count_all[key] += 1

            # Store question, answer, if part of balanced set, and if IDEFICs answer was perfect match
            all_data[b].append((question, ans, balanced, int(ans==pred_val), correct))

    print(f'Accuracy binary: {count_correct["bin"]/count_all["bin"]} ({count_correct["bin"]}/{count_all["bin"]})') 
    print(f'Accuracy open: {count_correct["open"]/count_all["open"]} ({count_correct["open"]}/{count_all["open"]})') 
    
    print(f'Perfect Match Accuracy binary: {perf_match["bin"]/count_all["bin"]} ({perf_match["bin"]}/{count_all["bin"]})')
    print(f'Perfect Match Accuracy open: {perf_match["open"]/count_all["open"]} ({perf_match["open"]}/{count_all["open"]})')

    if create_file:
        # Create a CSV comparing pred and gt_answer
        df = {'gt_answer':[], 'pred':[], 'question':[], 'balanced':[], 'perfect_match':[], 'correct':[]}

        for b in all_data:
            for i in range(len(all_data[b])):
                df['question'].append(all_data[b][i][0])
                df['gt_answer'].append(all_data[b][i][1])
                df['pred'].append(pred[b][i])  
                df['balanced'].append(all_data[b][i][2])
                df['perfect_match'].append(all_data[b][i][3])
                df['correct'].append(all_data[b][i][4])

        df = pd.DataFrame(df)
        df.to_csv(os.path.join(create_file,'err_analysis.csv'), index=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--exp_num', type=int, help='Experiment number for logging')
    parser.add_argument('--batchlim', type=float, default=float('inf'), help='Experiment number for logging')
    args = parser.parse_args()

    enum = args.exp_num
    batchlim = args.batchlim
    # assert sanity_check('Experiments/Exp2/raw_outputs.json', 'Experiments/Exp2/prompts_batched.json')
    # assert sanity_check(f'Experiments/Exp{enum}/raw_outputs.json', f'Experiments/Exp{enum}/prompts_batched.json')

    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('wordnet')
    answers = parse_answer(f'Experiments/Exp{enum}/raw_outputs.json', f'Experiments/Exp{enum}/answers.json')
    print(f'Exp{enum}')
    print(f'Batchsize: {len(answers["0"])}')
    # compute_accuracy('Experiments/Exp2/answers.json', 'Experiments/Exp2/qid_batched.json', create_file='Experiments/Exp2/')
    compute_accuracy(f'Experiments/Exp{enum}/answers.json', f'Experiments/Exp{enum}/qid_batched.json', create_file=f'Experiments/Exp{enum}/', batchlim=batchlim)
    # compute_accuracy_by_answer_type(f'Experiments/Exp{enum}/answers.json', f'Experiments/Exp{enum}/qid_batched.json', create_file=f'Experiments/Exp{enum}/',batchlim=batchlim)
