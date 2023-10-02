### Topic: Mental Tweet Train by GPT3
### Tag: [GPT3] [Train] [Tweets]
### Author: Wei-Hung, Tseng
### CreateDate: 2023/02/10
### SSH: ssh -p 22222 andy@ai2.glis.ntnu.edu.tw
### Conda env: mentalTweet
### Install:
'''
    pip install transformers simpletransformers emoji==0.6.0 openpyxl

    https://pytorch.org/get-started/locally/ (stable(1.12.0), Linux, Conda, Python, CUDA11.6)
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

    pip install openai
'''

### Execute:
'''
    python B3_train_GPT3.py
    nohup python B3_train_GPT3.py &
'''

### Reference:
'''
    How To Create A Custom Fine-Tuned Prediction Model Using Base GPT-3 models:
        https://cobusgreyling.medium.com/how-to-create-a-custom-fine-tuned-prediction-model-using-base-gpt-3-models-3dfd1eb1de0e

    Finetuning GPT3 With OpenAI:
        https://pakodas.substack.com/p/finetuning-gpt3-with-openai

    Openai Fine-tuning:
        https://platform.openai.com/docs/guides/fine-tuning

    Openai API Error Code Guidance:
        https://help.openai.com/en/articles/6891839-api-error-code-guidance
'''

import os, sys, time, re
import pandas, numpy
import openai
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

### Global variable
sTotaltime = time.time()
features = ["M", "R", "S", "HA", "HE"]
data = ['1017-M面向檢核後訓練資料', '1017-R面向檢核後訓練資料', '1017-S面向檢核後訓練資料', '0313-HA面向訓練資料更新', '1017-HE面向檢核後訓練資料']
cols = ["M1", "M5", "M10", "R1", "R2", "S1", "S2", "HA1", "HA2", "HE1", "HE2"]
dataDate = '20230324'
# col = 'M10'
col = sys.argv[1]
# print(f'col = {col}, sys.argv = {sys.argv}'); exit()
resultFilePath = f'./B3_data/result/{dataDate}/raw_{col}.txt'

### My case
def create_train_test_data():
    for feature, data1 in [(features[i], data[i]) for i in range(0, len(features))]:
        source_path = f'./A_data/20230313/{data1}.xlsx'
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
        f'./A_data/20230207/0207-twitter新增0分範例.xlsx',
        '0分範例', ## Work sheet
        names = ["tweet"],
        usecols = 'C'
    )

    lst = []
    for ind, row in M_df.iterrows(): lst.append([str(row['tweet']), row['M1']+2, row['M5']+2, row['M10']+2] + [2]*8)
    for ind, row in R_df.iterrows(): lst.append([str(row['tweet'])] + [2]*3 + [row['R1']+2, row['R2']+2] + [2]*6)
    for ind, row in S_df.iterrows(): lst.append([str(row['tweet'])] + [2]*5 + [row['S1']+2, row['S2']+2] + [2]*4)
    for ind, row in HA_df.iterrows(): lst.append([str(row['tweet'])] + [2]*7 + [row['HA1']+2, row['HA2']+2] + [2]*2)
    for ind, row in HE_df.iterrows(): lst.append([str(row['tweet'])] + [2]*9 + [row['HE1']+2, row['HE2']+2])
    for ind, row in zero_df.iterrows(): lst.append([str(row['tweet'])] + [2]*11)
    df = pandas.DataFrame(lst, columns = (["tweet"] + cols))
    del M_df; del R_df; del S_df; del HA_df; del HE_df; del zero_df
    # print(len(df)); print()
    # print(df.head()); print()
    # exit()

    for i, col in enumerate(cols):
        # df_tmp.rename(columns = {'tweet':'prompt', f'{col}':'completion'}, inplace = True)
        text_df = df.iloc[:,[0]]
        labels_df = df.iloc[:,[(i+1)]]
        # print(text_df); print(labels_df); exit()
        X_train, y_train, X_test, y_test = train_test_split(text_df, labels_df, test_size = 0.2, stratify = labels_df)
        # print(X_train[:5]); print(y_train[:5]); print(X_test[:5]); print(y_test[:5]); exit()
        train_df = pandas.DataFrame(numpy.concatenate((X_train, X_test), axis = 1), columns =['prompt', 'completion'])
        test_df = pandas.DataFrame(numpy.concatenate((y_train, y_test), axis = 1), columns =['prompt', 'completion'])
        # print(train_df); print(test_df); exit()
        train_df.to_json(f"./B3_data/source/{dataDate}/{col}_train_08.jsonl", orient='records', lines=True)
        test_df.to_csv(f"./B3_data/source/{dataDate}/{col}_test_02.csv", index=False)

## Now the OpenAI utility can be used to analyse the JSONL file.
'''
    In colab notebook:
    !openai tools fine_tunes.prepare_data -f vehicles.jsonl -q

    In terminal:
    openai tools fine_tunes.prepare_data -f ./B3_data/M1_train_08.jsonl -q
    openai tools fine_tunes.prepare_data -f ./B3_data/M5_train_08.jsonl -q
    openai tools fine_tunes.prepare_data -f ./B3_data/M10_train_08.jsonl -q
    openai tools fine_tunes.prepare_data -f ./B3_data/R1_train_08.jsonl -q
    openai tools fine_tunes.prepare_data -f ./B3_data/R2_train_08.jsonl -q
    openai tools fine_tunes.prepare_data -f ./B3_data/S1_train_08.jsonl -q
    openai tools fine_tunes.prepare_data -f ./B3_data/S2_train_08.jsonl -q
    openai tools fine_tunes.prepare_data -f ./B3_data/HA1_train_08.jsonl -q
    openai tools fine_tunes.prepare_data -f ./B3_data/HA2_train_08.jsonl -q
    openai tools fine_tunes.prepare_data -f ./B3_data/HE1_train_08.jsonl -q
    openai tools fine_tunes.prepare_data -f ./B3_data/HE2_train_08.jsonl -q

    dynamic:
        openai tools fine_tunes.prepare_data -f ./B3_data/source/20230317/HE2_train_08.jsonl -q
'''

## Now we can start the training process and from this point an OpenAI api key is required.
'''
    !openai --api-key 'xxxxxxxxxxxxxxxxx' api fine_tunes.create -t "vehicles_prepared_train.jsonl" -v "vehicles_prepared_valid.jsonl" --compute_classification_metrics --classification_positive_class " autos" -m ada
    openai api fine_tunes.create -t test.jsonl -m ada --suffix "custom model name"

    M1: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/M1_train_08_prepared_train.jsonl" -v "./B3_data/M1_train_08_prepared_valid.jsonl" \
        -m ada --suffix "M1" --compute_classification_metrics --classification_n_classes 5

    M5: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/M5_train_08_prepared_train.jsonl" -v "./B3_data/M5_train_08_prepared_valid.jsonl" \
        -m ada --suffix "M5" --compute_classification_metrics --classification_n_classes 5

    M10: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/M10_train_08_prepared_train.jsonl" -v "./B3_data/M10_train_08_prepared_valid.jsonl" \
        -m ada --suffix "M10" --compute_classification_metrics --classification_n_classes 5

    R1: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/R1_train_08_prepared_train.jsonl" -v "./B3_data/R1_train_08_prepared_valid.jsonl" \
        -m ada --suffix "R1" --compute_classification_metrics --classification_n_classes 5
    
    R2: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/R2_train_08_prepared_train.jsonl" -v "./B3_data/R2_train_08_prepared_valid.jsonl" \
        -m ada --suffix "R2" --compute_classification_metrics --classification_n_classes 5
    
    S1: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/S1_train_08_prepared_train.jsonl" -v "./B3_data/S1_train_08_prepared_valid.jsonl" \
        -m ada --suffix "S1" --compute_classification_metrics --classification_n_classes 5
    
    S2: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/S2_train_08_prepared_train.jsonl" -v "./B3_data/S2_train_08_prepared_valid.jsonl" \
        -m ada --suffix "S2" --compute_classification_metrics --classification_n_classes 5
    
    HA1: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/HA1_train_08_prepared_train.jsonl" -v "./B3_data/HA1_train_08_prepared_valid.jsonl" \
        -m ada --suffix "HA1" --compute_classification_metrics --classification_n_classes 5
    
    HA2: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/HA2_train_08_prepared_train.jsonl" -v "./B3_data/HA2_train_08_prepared_valid.jsonl" \
        -m ada --suffix "HA2" --compute_classification_metrics --classification_n_classes 5
    
    HE1: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/HE1_train_08_prepared_train.jsonl" -v "./B3_data/HE1_train_08_prepared_valid.jsonl" \
        -m ada --suffix "HE1" --compute_classification_metrics --classification_n_classes 5
    
    HE2: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/HE2_train_08_prepared_train.jsonl" -v "./B3_data/HE2_train_08_prepared_valid.jsonl" \
        -m ada --suffix "HE2" --compute_classification_metrics --classification_n_classes 5
    
    dynamic: 
        openai --api-key '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' api fine_tunes.create \
        -t "./B3_data/source/20230317/HE2_train_08_prepared_train.jsonl" \
        -v "./B3_data/source/20230317/HE2_train_08_prepared_valid.jsonl" \
        -m ada --suffix "HE2" --compute_classification_metrics --classification_n_classes 5

        (mentalTweet) andy@server:/nfs/andy/mental$ 
        
        openai --api-key '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' api fine_tunes.follow -i \
        ft-7HPYfPtywqToeRp4QxT6hSt0
        
        [2023-02-17 21:29:25] Created fine-tune: ft-YwHESpCf7yNNNzgt7WuPLtrJ
        [2023-02-17 21:36:35] Fine-tune costs $0.21
        [2023-02-17 21:36:36] Fine-tune enqueued. Queue number: 0
        [2023-02-17 21:36:40] Fine-tune started
        [2023-02-17 21:41:11] Completed epoch 1/4
        [2023-02-17 21:45:41] Completed epoch 2/4
        [2023-02-17 21:50:10] Completed epoch 3/4
        [2023-02-17 21:54:40] Completed epoch 4/4
        [2023-02-17 21:55:11] Uploaded model: ada:ft-personal-2023-02-17-13-55-11
        [2023-02-17 21:55:12] Uploaded result file: file-2NxM9rQL0ejNMHjTaaPLIn0W
        [2023-02-17 21:55:12] Fine-tune succeeded

        Job complete! Status: succeeded 🎉
        Try out your fine-tuned model:

        openai api completions.create -m ada:ft-personal-2023-02-17-13-55-11 -p <YOUR_PROMPT>

'''

### Predict
## Create train & test data
# create_train_test_data(); exit()

## api_key create at 2023年2月17日
openai.api_key = "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"
## ft_model: In output of the training process
## 20230306 8:2 model
# ft_model = 'ada:ft-personal:m1-2023-03-02-03-27-44'
# ft_model = 'ada:ft-personal:m5-2023-03-03-06-45-02'
# ft_model = 'ada:ft-personal:m10-2023-03-03-08-51-52'
# ft_model = 'ada:ft-personal:r1-2023-03-04-05-52-34'
# ft_model = 'ada:ft-personal:r2-2023-03-04-09-30-33'
# ft_model = 'ada:ft-personal:s1-2023-03-04-11-43-59'
# ft_model = 'ada:ft-personal:s2-2023-03-04-16-29-44'
# ft_model = 'ada:ft-personal:ha1-2023-03-05-06-54-06'
# ft_model = 'ada:ft-personal:ha2-2023-03-05-09-07-29'
# ft_model = 'ada:ft-personal:he1-2023-03-05-16-50-00'
# ft_model = 'ada:ft-personal:he2-2023-03-06-02-50-02'

## 20230317 8:2 model
if col == 'M1': ft_model = 'ada:ft-personal:m1-2023-03-17-02-46-32'
if col == 'M5': ft_model = 'ada:ft-personal:m5-2023-03-17-08-10-55'
if col == 'M10': ft_model = 'ada:ft-personal:m10-2023-03-17-16-50-08'
if col == 'R1': ft_model = 'ada:ft-personal:r1-2023-03-18-06-33-57'
if col == 'R2': ft_model = 'ada:ft-personal:r2-2023-03-18-10-58-19'
if col == 'S1': ft_model = 'ada:ft-personal:s1-2023-03-18-13-23-03'
if col == 'S2': ft_model = 'ada:ft-personal:s2-2023-03-19-12-28-54'
if col == 'HA1': ft_model = 'ada:ft-personal:ha1-2023-03-19-16-52-10'
if col == 'HA2': ft_model = 'ada:ft-personal:ha2-2023-03-20-03-58-52'
if col == 'HE1': ft_model = 'ada:ft-personal:he1-2023-03-20-08-06-32'
if col == 'HE2': ft_model = 'ada:ft-personal:he2-2023-03-20-10-20-19'

## Read data: for train
# df = pandas.read_csv(f'./B3_data/source/{dataDate}/{col}_test_02.csv', lineterminator='\n')
# tweet_df = df.iloc[:,[0]]
# ans_df = df.iloc[:,[1]]

## Read data: for predict
df = pandas.read_csv(f'./predict_result/data{dataDate}/del_part_zero_df.csv', lineterminator='\n')
tweet_df = df.iloc[:,[1]]
tweet_df.columns = ['prompt']
# print(tweet_df.head()); exit()

predict_lst = []
# carryOn = 0
carryOn = int(sys.argv[2])
print(col)
for ind, row in tweet_df.iterrows():
    print(ind)
    if ind < carryOn: continue
    ## temperature: randomness
    res = openai.Completion.create(
        model=ft_model, 
        prompt=row['prompt'] + '\n\n###\n\n', 
        max_tokens=1, 
        temperature=0, 
        logprobs=5
    )
    predictResult = int(res['choices'][0]['text'])
    predict_lst.append(predictResult)
    f = open(resultFilePath, "a")
    f.write(f'predict = {predictResult}, ind = {ind}\n')
    f.close()
    ## Limit: 60.000000 / min
    # time.sleep(1) ## 41:RateLimitError
    # time.sleep(3) ## 322:RateLimitError
    # time.sleep(5) ## 2622:RateLimitError
    time.sleep(7) ## 500 {'error': {'message': 'The server had an error while processing your request. Sorry about that! You can retry your request, or contact us through our help center at help.openai.com if the error persists. (Please include the request ID 371d046654337518a7133342b7e72e96 in your message.)', 'type': 'server_error', 'param': None, 'code': None}} {'Date': 'Thu, 23 Feb 2023 20:08:13 GMT', 'Content-Type': 'application/json', 'Content-Length': '366', 'Connection': 'keep-alive', 'Access-Control-Allow-Origin': '*', 'Openai-Processing-Ms': '15146', 'Openai-Version': '2020-10-01', 'Strict-Transport-Security': 'max-age=15724800; includeSubDomains', 'X-Request-Id': '371d046654337518a7133342b7e72e96'}

f = open(resultFilePath, "a")
f.write("\nTotal time: %1.2f\n"%(time.time()-sTotaltime))
f.close()
print('finish')
## M5: [1] 4169992
## M10: 715994
## nohup python B3_train_predict_GPT3.py M10 7679 > raw_M10_stdout.txt 2> raw_M10_stderr.txt &
## R1: 872565
## nohup python B3_train_predict_GPT3.py R1 0 > raw_R1_stdout.txt 2> raw_R1_stderr.txt &
## R2: 1639485
## nohup python B3_train_predict_GPT3.py R2 0 > raw_R2_stdout.txt 2> raw_R2_stderr.txt &
## S1: 2047769
## nohup python B3_train_predict_GPT3.py S1 2642 > raw_S1_stdout.txt 2> raw_S1_stderr.txt &
## S2: 2345002
## nohup python B3_train_predict_GPT3.py S2 0 > raw_S2_stdout.txt 2> raw_S2_stderr.txt &