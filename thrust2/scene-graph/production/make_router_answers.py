import csv
import pandas as pd
import json
import re
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# pattern = re.compile(r'Answer:\s*(\w+)')

def format_answer(answer):
    # ans = None
    # answer = answer.split()
    # for i in range(len(answer)):
    #     if 'Answer:' in answer[i]:
    #         ans = answer[i].replace('Answer:','')
    #         if ans=='' and i<len(answer):
    #             ans = answer[i+1]
    #         break

    # print('??', ans)
    # ans = re.findall(r'Answer:(.*)', answer)
    # ans = re.findall(r'Answer:\s*(\w+)',answer)
    ans = re.findall(r'Answer:(.*?)(?:\.|Question:|\n|$)',answer)
    # answer = answer.replace('\n','')
    # match = pattern.search(answer)
    # ans = match.group(1)
    # ans = pattern.findall(answer)
    if not ans:
        return ''
    ans = ans[0]

    # Clean answer:
    ans = ans.lower()
    
    # Remove punctuation:
    ans = re.sub(r'[^\w\s]','',ans)
    # print(ans)
    # Remove stopwords using nltk:
    # stop_words = set(nltk.corpus.stopwords.words('english'))
    # word_tokens = nltk.word_tokenize(ans)
    # ans = [w for w in word_tokens if w not in stop_words]
    # print(ans)

    # Make singular
    # lemmatizer = nltk.stem.WordNetLemmatizer()
    # ans = [lemmatizer.lemmatize(word, pos='n') for word in ans]
    # print(ans)
    # ans = ' '.join(ans)
    ans = ans.strip()
    return ans

with open('raw_outputs_pragnyas_final_finetuned_test.json') as f:
# with open('raw_outputs_pragnyas_final_train.json') as f:
# with open('raw_outputs_pragnyas.json') as f:
    answers = json.load(f)
    answers = answers['answers']
    # r_answers = answers
    answers = [format_answer(x) for x in answers]
    # print(len(answers))
    # for i in range(20):
    #     print(r_answers[i])
    #     print('>>',format_answer(r_answers[i]))
    #     print('\n\n')


# df = pd.read_csv('data/router_train.csv')
df = pd.read_csv('data/mqa_test_balanced.csv')
# df = df.iloc [:47724]
df['T2_answer'] = answers

df.to_csv('data/router_test_answered_pragnyas_final_finetuned.csv',index=False)
# df.to_csv('data/router_train_answered_pragnyas_final_finetuned.csv',index=False)

# df = pd.read_csv('data/router_train_small_answered_pragnyas.csv')
full = len(df)
print(set(df['sample_dataset']))
df = df[df['sample_dataset']=='gqa']
# df = df[df['sample_dataset']=='tallyqa']
print('Proportion of GQA:',len(df)/full)

correct = 0
for i,row in df.iterrows():
    # if (row['answer']==row['T2_answer']):
    if ((row['answer']==row['T2_answer']) or 
        (str(row['answer']) in str(row['T2_answer'])) or 
        (str(row['T2_answer']) in str(row['answer']))):
        if row['T2_answer']=='':
            continue
        correct += 1

print(correct/len(df))
print(correct,'/',len(df))