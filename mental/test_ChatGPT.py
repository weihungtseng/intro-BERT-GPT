# Note: you need to be using OpenAI Python v0.27.0 for the ChatGPT below to work
import openai
import os
from pprint import pprint

# 设置 API 密钥
## Step 1
# $ export OPENAI_API_KEY=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
## Step 2
# $ python test_ChatGPT.py

## Method 1
# $ export OPENAI_API_KEY=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
# openai.api_key = os.environ["OPENAI_API_KEY"]

## Method 2
openai.api_key = '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'
# openai.api_key = os.environ["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"] (wrong)


# # 使用 GPT-3 模型生成文本
# model_engine = "text-davinci-002"

# prompt = '''
#     Please play the role as a System and I will play the role as a User.
#     System:
#     You are a tutor that always responds in the Socratic style. 
#     You *never* give the student the answer, but always try to ask just the right question 
#     to help them learn to think for themselves. 
#     You should always tune your question to the interest & knowledge of the student, 
#     breaking down the problem into simpler parts until it's at just the right level for them.

#     User:
#     How do I solve the system of linear equations: 3x + 2y = 7, 9x -4y = 1.

#     System:
# '''

# # prompt = '''
# #     請幫我以正體中文寫出校慶典禮致詞稿。
# # '''

# # below is for GPT-3
# response = openai.Completion.create(
#     engine=model_engine, 
#     prompt=prompt, 
#     temperature=1, # 0.0 ~ 2.0
#     max_tokens=100
# )
# print("GPT-3: ", response.choices[0].text.strip())
# pprint(response)
# # exit()
# print("\n-------End of GPT-3 ----------\n")


# # Below is for ChatGPT
# response = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#         {"role": "system",    "content": "You are a helpful assistant."},
#         {"role": "user",      "content": "誰贏了世界杯 World Series 2020?"},
#         {"role": "assistant", "content": "Los Angeles Dodgers 贏了 World Series 2020."},
#         {"role": "user",      "content": "在哪裡舉行?"}
#     ]
# )
# print("ChatGPT: ")
# pprint(response)
# print("\n-------End of calling ChatGPT version 1----------\n")


# import requests
# response = requests.post(
#     'https://api.openai.com/v1/chat/completions',
#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': f'Bearer {openai.api_key}'
#     },
#     json = {
#         'model': 'gpt-3.5-turbo', # 一定要用chat可以用的模型
#         'messages' : [{"role": "user", "content": prompt}]
#     })

# # 使用json解析
# json = response.json()
# pprint(json)
# print("\n-------End of calling ChatGPT version 2----------\n")

## My implement
import pandas
import numpy
import re
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

features = ["M", "R", "S", "HA", "HE"]
data = ['1017-M面向檢核後訓練資料', '1017-R面向檢核後訓練資料', '1017-S面向檢核後訓練資料', '0313-HA面向訓練資料更新', '1017-HE面向檢核後訓練資料']
cols = ["M1", "M5", "M10", "R1", "R2", "S1", "S2", "HA1", "HA2", "HE1", "HE2"]

def tcfunc(x, n=4): ## trancate a number to have n decimal digits
    d = '0' * n
    d = int('1' + d)
    if isinstance(x, (int, float)): return int(x * d) / d
    return x

def show_Result(predictions, test_yL, scoreResultPath):
    f = open(scoreResultPath, "a")
    f.write('MicroF1, MacroF1 = %0.4f\t%0.4f\n\n' %
        (metrics.f1_score(test_yL, predictions, average='micro'),
        metrics.f1_score(test_yL, predictions, average='macro')))

    f.write('\t\tPrecision\tRecall\t\tF1\t\t\tSupport\n')

    (Precision, Recall, F1, Support) = list(map(tcfunc, 
        precision_recall_fscore_support(test_yL, predictions, average='micro')))
    f.write('Micro\t{}\t\t{}\t\t{}\t\t{}\n'.format(Precision, Recall, F1, Support))

    (Precision, Recall, F1, Support) = list(map(tcfunc, 
        precision_recall_fscore_support(test_yL, predictions, average='macro'))) 
    f.write('Macro\t{}\t\t{}\t\t{}\t\t{}\n\n'.format(Precision, Recall, F1, Support))

    f.write('Test accuracy is %1.4f\n\n'%(accuracy_score(test_yL, predictions)))
    f.write(classification_report(test_yL, predictions))
    f.write('\n\n')
    f.close()

def makeData():
    for feature, data1 in [(features[i], data[i]) for i in range(0, len(features))]:
        source_path = f'../../data/A_k_fold_cross_validation/data/20230313/{data1}.xlsx'
        ## Read data from excel(file, worksheet, col name, col index)
        if(feature == "M"):
            M_df = pandas.read_excel(
                source_path,
                'M', ## Work sheet
                names = ["tweet", "M1", "M5", "M10"],
                usecols = 'B:E'
            )
        if(feature == "R"):
            R_df = pandas.read_excel(
                source_path,
                'R',
                names = ["tweet", "R1", "R2"],
                usecols = 'B:D'
            )
        if(feature == "S"):
            S_df = pandas.read_excel(
                source_path,
                'S',
                names = ["tweet", "S1", "S2"],
                usecols = 'B:D'
            )
        if(feature == "HA"):
            HA_df = pandas.read_excel(
                source_path,
                'HA',
                names = ["tweet", "HA1", "HA2"],
                usecols = 'B:D'
            )
        if(feature == "HE"):
            HE_df = pandas.read_excel(
                source_path,
                'HE',
                names = ["tweet", "HE1", "HE2"],
                usecols = 'B:D'
            )

    zero_df = pandas.read_excel(
        f'../../data/A_k_fold_cross_validation/data/20230207/0207-twitter新增0分範例.xlsx',
        '0分範例', ## Work sheet
        names = ["tweet"],
        usecols = 'C'
    )

    lst = []; i = 0
    for ind, row in M_df.iterrows(): lst.append([i ,str(row['tweet']), row['M1'], row['M5'], row['M10']] + [0]*8); i += 1
    for ind, row in R_df.iterrows(): lst.append([i ,str(row['tweet'])] + [0]*3 + [row['R1'], row['R2']] + [0]*6); i += 1
    for ind, row in S_df.iterrows(): lst.append([i ,str(row['tweet'])] + [0]*5 + [row['S1'], row['S2']] + [0]*4); i += 1
    for ind, row in HA_df.iterrows(): lst.append([i ,str(row['tweet'])] + [0]*7 + [row['HA1'], row['HA2']] + [0]*2); i += 1
    for ind, row in HE_df.iterrows(): lst.append([i ,str(row['tweet'])] + [0]*9 + [row['HE1'], row['HE2']]); i += 1
    for ind, row in zero_df.iterrows(): lst.append([i ,str(row['tweet'])] + [0]*11); i += 1
    df = pandas.DataFrame(lst, columns = (['id', 'tweet'] + cols))
    del lst; del M_df; del R_df; del S_df; del HA_df; del HE_df; del zero_df
    # print(df.head()); print()

    trainSize = 10
    for i, col in enumerate(cols):
        tmp_df = df.iloc[:,[0,1,(i+2)]]
        tmp_df.columns = ['id', 'tweet', 'label']
        tmp_df.to_csv(f'../../data/test_ChatGPT/data/standard/{col}_standard.csv', index=False)
        train_lst = []; test_lst = []
        cnt1 = 0; cnt2 = 0; cnt3 = 0; cnt4 = 0; cnt5 = 0; 
        for ind, row in tmp_df.iterrows():
            if row['label'] == -2 and cnt1 != trainSize: train_lst.append([row['id'], row['tweet'], row['label']]); cnt1 += 1
            elif row['label'] == -1 and cnt2 != trainSize: train_lst.append([row['id'], row['tweet'], row['label']]); cnt2 += 1
            elif row['label'] == 0 and cnt3 != trainSize: train_lst.append([row['id'], row['tweet'], row['label']]); cnt3 += 1
            elif row['label'] == 1 and cnt4 != trainSize: train_lst.append([row['id'], row['tweet'], row['label']]); cnt4 += 1
            elif row['label'] == 2 and cnt5 != trainSize: train_lst.append([row['id'], row['tweet'], row['label']]); cnt5 += 1
            else: test_lst.append([row['id'], row['tweet'], row['label']])
        train_df = pandas.DataFrame(train_lst, columns = (['id', 'tweet', 'label']))
        test_df = pandas.DataFrame(test_lst, columns = (['id', 'tweet', 'label']))
        train_df.to_csv(f'../../data/test_ChatGPT/data/standard/{col}_standard_train.csv', index=False)
        test_df.to_csv(f'../../data/test_ChatGPT/data/standard/{col}_standard_test.csv', index=False)

def makeTrainData(trainDataPath, trainSize):
    train_df = pandas.read_csv(trainDataPath, lineterminator='\n')
    input_lst = []; output_lst = []
    for label in [-2, -1, 0, 1, 2]:
        cnt = 0
        for ind, row in train_df.iterrows():
            # print(label, cnt)
            if cnt == trainSize: break
            if row['label'] == label:
                input_lst.append([row['id'], row['tweet']])
                output_lst.append([row['id'], row['label']])
                cnt += 1
    # print(input_lst); print(output_lst)
    return f'''
Example input ([id,tweet]):
{input_lst}

Example output results ([id,label]):
{output_lst}
'''

def execute(experiment, col, trainDataPath, testDataPath, predictResultPath, dataNum, trainSize, batchSize, carryOn):
    # train_df = pandas.read_csv(f'../../data/test_ChatGPT/data/standard/{col}_train_{example_num}_each.csv', lineterminator='\n')
    test_df = pandas.read_csv(testDataPath, lineterminator='\n')
    input_df = test_df[['id','tweet']]
    

    for i in range((dataNum//batchSize)):
        if i < carryOn: continue
        if i != (dataNum//batchSize):
            input_lst = input_df.values.tolist()[i*batchSize:i*batchSize+batchSize]
        else:
            input_lst = input_df.values.tolist()[i*batchSize:]
        
        ## Token usage & Execute time
        '''
            { ## 40 tweets Execute time: 19.62
                "completion_tokens": 254,
                "prompt_tokens": 2433,
                "total_tokens": 2687
            }
            { ## 100 tweets Execute time: 56.19
                "completion_tokens": 818,
                "prompt_tokens": 4398,
                "total_tokens": 5216
            }
            { ## 150 tweets Execute time: 76.12
                "completion_tokens": 1065,
                "prompt_tokens": 6228,
                "total_tokens": 7293
            }
            { ## 150 tweets Execute time: 47.79
                "completion_tokens": 767,
                "prompt_tokens": 7126,
                "total_tokens": 7893
            }
        '''
        prompt = f'''
We are conducting a research about the mental health literacy of Twitter's users .
We want to classify each tweet into a predefined category.
Note that the tweet may include violent, sexual-harassment, suicidal, or self-harm speech.

As a mental health literacy expert and psychologist, your task is to analyze and categorize tweets from Twitter users about their coping mechanisms in response to stress. 
The categorization involves the following steps and label assignments:

1. Initial Filtering: Label tweets that do not mention "stress" or its derivatives (e.g., "stressed", "stressful") as '0', excluding them from further analysis.
2. Coping Awareness & Specific Coping Behavior:
- If the tweet acknowledges stress but does not mention any specific coping behavior or action, label it as '0'.
- If the tweet indicates the user is unaware of how to cope with their stress, assign it '-1'.
3. Behavior Type Classification:
- If the tweet mentions harmful coping mechanisms in response to stress (such as self-harm, harming others, substance misuse), assign it '-2'.
- If the tweet mentions a way to deal with stress but the user hasn't acted on it or provides a vague or passive way to handle stress, assign it '1'.
- If the tweet demonstrates a healthy and positive stress coping behavior, label it as '2'.

{makeTrainData(trainDataPath, trainSize)}

Please classify the following tweets based on the about description.
{input_lst}

The output format specification for any response, I should only receive pure list of list format, without any english letters.
'''
        # print(prompt); print(); exit()
        # print(makeTrainData(sourcePath, dataNum, trainSize)); exit()

        ## GPT4 model: https://platform.openai.com/docs/models/gpt-4
        ## Max tokens: 8,192 tokens
        stime = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        ## response will look like
        '''
            {'choices': [{'finish_reason': 'stop',
                        'index': 0,
                        'message': {'content': '[[0, 1], [1, 2], [2, 1], [3, 2], [4, 1], '
                                                '[5, 2], [6, 2], [7, 1], [8, 2], [9, -1]]',
                                    'role': 'assistant'}}],
            'created': 1690694155,
            'id': 'chatcmpl-7ht4V72ejsMzcfbIkVIBR3hpdO2HI',
            'model': 'gpt-4-0613',
            'object': 'chat.completion',
            'usage': {'completion_tokens': 60,
                    'prompt_tokens': 1296,
                    'total_tokens': 1356}}

            print(response['choices'][0]['message']['content'])
            [[0, 1], [1, 2], [2, 1], [3, 2], [4, 1], [5, 2], [6, 2], [7, 1], [8, 2], [9, -1]]
        '''
        
        print(f"{response['usage']}\n")
        print(f"{response['choices'][0]['message']['content']}\n")
        print(f"col = {col}\n")
        print(f"trainSize = {trainSize}\n")
        print(f"batchSize = {batchSize}\n")
        print(f"i = {i}\n")
        print(f"\n\nExecute time: {(time.time()-stime):.2f}\n")

        ## response parse to df to csv
        pattern = r'\[.*?\]'
        # print(response['choices'][0]['message']['content']); exit()
        sub_lists = re.findall(pattern, response['choices'][0]['message']['content'].replace("[","",1))
        lst = [eval(sub_list) for sub_list in sub_lists]
        df = pandas.DataFrame(lst, columns = ['id', 'label'])
        if os.path.exists(predictResultPath):
            df.to_csv(predictResultPath, mode='a', index=False, header=False)
        else: 
            df.to_csv(predictResultPath, index=False)
        time.sleep(15)
        ## Problem
        '''
        openai.error.Timeout: Request timed out: HTTPSConnectionPool(host='api.openai.com', port=443): Read timed out. (read timeout=600)
        
        openai.error.RateLimitError: Rate limit reached for 10KTPM-200RPM in organization org-cZxQJmtGYK9MLRDulZ2g4ucH on tokens per min. Limit: 10000 / min. Please try again in 6ms. Contact us through our help center at help.openai.com if you continue to have issues.
        '''
    
    ## F1-Score
    print('Calculate F1-Score')
    scoreResultPath = f'../../data/test_ChatGPT/result/standard/{experiment}/gpt4_result_{experiment}.txt'

    df = pandas.read_csv(predictResultPath, lineterminator='\n')
    test_df = pandas.read_csv(testDataPath, lineterminator='\n')
    predict_se = df['label']#.apply(lambda x: x+2)
    answer_se = test_df['label']#.apply(lambda x: x+2)
    f = open(scoreResultPath, "a")
    f.write(f"col = {col}\n")
    f.write(f'experiment = {experiment}\n')
    f.write(f'dataNum = {dataNum}\n')
    f.write(f'trainSize = {trainSize}\n')
    f.write(f'batchSize = {batchSize}\n\n')
    f.close()
    show_Result(predict_se.tolist(), answer_se.tolist()[:dataNum], scoreResultPath)

def view(experiment, col, testDataPath, predictResultPath, dataNum, trainSize, batchSize):
    test_df = pandas.read_csv(testDataPath, lineterminator='\n').iloc[:dataNum]
    predict_df = pandas.read_csv(predictResultPath, lineterminator='\n')
    view_df = pandas.concat([test_df, predict_df['label']], axis = 1, ignore_index = True)
    view_df.columns = ['id', 'tweet', 'label', 'predict']
    view_df.to_csv(f'../../data/test_ChatGPT/result/standard/{experiment}/view_{col}_{experiment}_{trainSize}_{batchSize}.csv', index=False)

if __name__ == "__main__":
    # makeData(); exit()
    ## totalNum = 3377

    ## ex1
    # experiment = 'ex1'
    # dataNum = 500
    # batchSize = 10
    # execute(experiment, dataNum, batchSize, carryOn=0)
    # f1Score(experiment, dataNum, batchSize)
    # view(dataNum)

    ## ex2
    # ex2ExecuteTime = time.time()
    # experiment = 'ex2'
    # dataNum = 600
    # batchSizes = [
    #     # (1,0),(5,0),(10,39),(20,0),(30,0),(40,0),(50,0),(100,0),
    #     (150,3)
    # ]
    # for batchSize, carryOn in batchSizes:
    #     execute(experiment, dataNum, batchSize, carryOn)
    #     f1Score(experiment, dataNum, batchSize)
    #     # view(dataNum)
    # print(f"\nEx2 execute time: {(time.time()-ex2ExecuteTime):.2f}\n")

    ## ex3
    ex3ExecuteTime = time.time()
    experiment = 'ex3'
    dataNum = 600
    trainSizes = [
        # 1,5,
        10
    ]
    batchSizes = [
        (1,0),(5,0),(10,0),(20,0),(30,0),(40,0),(50,0),(100,0)
    ]
    for col in cols:
        sourcePath = f'../../data/test_ChatGPT/data/standard/{col}_standard.csv'
        trainDataPath = f'../../data/test_ChatGPT/data/standard/{col}_standard_train.csv'
        testDataPath = f'../../data/test_ChatGPT/data/standard/{col}_standard_test.csv'
        for trainSize in trainSizes:
            for batchSize, carryOn in batchSizes:
                predictResultPath = f'../../data/test_ChatGPT/result/standard/{experiment}/{col}_{experiment}_{trainSize}_{batchSize}.csv'
                # view(
                #     experiment, 
                #     col, 
                #     testDataPath, 
                #     predictResultPath, 
                #     dataNum, 
                #     trainSize, 
                #     batchSize
                # ); exit()
                execute(
                    experiment, 
                    col, 
                    trainDataPath, 
                    testDataPath, 
                    predictResultPath, 
                    dataNum, 
                    trainSize, 
                    batchSize, 
                    carryOn
                )
        break
    print(f"\nEx3 execute time: {(time.time()-ex3ExecuteTime):.2f}\n")

## prompt example
'''
prompt = f''
    Here is a research question on text classification for you: 
    This study explores the correlations among five psychological dimensions in 
    Twitter users' posts, namely M, R, S, HA, and HE. 
    These five dimensions are further divided into eleven indicators: 
    M1, M5, M10, R1, R2, S1, S2, HA1, HA2, HE1, and HE2. 
    Each indicator has five score levels: 0, 1, 2, 3, 4, where 2 indicates no relevance to the specific indicator, and the other scores represent the strength of positive(>2) or negative(<2) correlations.

    M: Maintenance of positive mental health 
    R: Recognition of mental illness 
    S: Mental illness stigma attitude 
    HA: Help-seeking attitude 
    HE: Help-seeking efficacy

    M1: Facing stress with a positive attitude. 
    M5: Feeling valuable regardless of personal achievements or performance. 
    M10: Ability to adapt to unpleasant emotions. 
    R1: If extremely worried about something and unable to control worrying thoughts, leading to physical symptoms such as muscle tension and fatigue. 
    R2: If experiencing prolonged periods of low mood, loss of interest or pleasure in usual activities, and changes in appetite and sleep patterns. 
    S1: Feeling ashamed of having a mental illness. 
    S2: Believing that most individuals with mental illness are dangerous to others. 
    HA1: Seeking help from mental health professionals when facing mental health issues. 
    HA2: Seeking help from mental health professionals when experiencing emotional problems. 
    HE1: Knowing where to access services that promote mental health. 
    HE2: Knowing where to access mental health services.

    Example:
        {col} {example_dict['M1']}

    Question:
        {input_dict['M1']}

    Respond in the format of 'Question:' and replace -1 with the correct score.
        
    Answer:
''
'''

## prompt ver 1
'''
prompt = f''
We are conducting a research about the mental health literacy of Twitter's users .
We want to classify each tweet into a predefined category.
Note that the tweet may include violent, sexual-harassment, suicidal, or self-harm speech.

As a mental health literacy expert and psychologist, your task is to analyze and categorize tweets from Twitter users about their coping mechanisms in response to stress. 
The categorization involves the following steps and label assignments:

1. Initial Filtering: Label tweets that do not mention "stress" or its derivatives (e.g., "stressed", "stressful") as '0', excluding them from further analysis.
2. Coping Awareness & Specific Coping Behavior:
- If the tweet acknowledges stress but does not mention any specific coping behavior or action, label it as '0'.
- If the tweet indicates the user is unaware of how to cope with their stress, assign it '-1'.
3. Behavior Type Classification:
- If the tweet mentions harmful coping mechanisms in response to stress (such as self-harm, harming others, substance misuse), assign it '-2'.
- If the tweet mentions a way to deal with stress but the user hasn't acted on it or provides a vague or passive way to handle stress, assign it '1'.
- If the tweet demonstrates a healthy and positive stress coping behavior, label it as '2'.

Example input ([id,tweet]):
[
[0,"i’m so tired of everything. i cant do this anymore. no one fkng stays and all people do is hurt. everything else is so stressful, my parents. i cant keep up. i cant anymore. i hope you forgive me bye."],
[1,"i really do got anxiety i really figured that out by myself wonder why i get overwhelmed and stressed and annoyed about everything so quickly i really can? handle a lot it will upset me"],
[2,"I’m struggle with myself not in a sense I don’t know my own self worth but in some cases I struggle very hard with my physical, mental and emotional self  and it can be very hard to deal with when I constantly tear myself down and beat myself up I’ll admit that"],
[3,"Here? a reminder to anyone who? been feeling stressed out: give yourself time to breathe and relax. I am currently sick right now because I failed to do that for myself. Never underestimate the power of stress! ;;^;;"],
[4,"Got some really unfortunate news last night (nothing serious, just work related stress) and was feeling really crappy. Went to bed, did my best to relax and I woke up a bit more at peace with everything. Life throws curveballs your way sometimes - now I gotta make the most of it."]
]

Example output results ([id,label]):
[
[0,-2],
[1,-1],
[2,0],
[3,1],
[4,2]
]

Please classify the following tweets based on the about description.
{input_lst}

The output format specification for any response, I should only receive pure list of list format, without any english letters.
''
'''

'''
研究限制：推文面向所有人，若是由專業人士撰寫推文比較準確
'''