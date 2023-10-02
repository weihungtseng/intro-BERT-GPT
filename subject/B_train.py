# Topic: NTNU 1-1 Final - Train
# Tag: [Train] [NTNU Library Data]
# Author: Wei-Hung, Tseng
# CreateDate: 2022/11/25
# SSH: 
# Conda env: include in [mentalTweet]
# Install: 
#   pip install transformers simpletransformers emoji==0.6.0 openpyxl tabulate

#   transformers simpletransformers emoji==0.6.0 openpyxl: deep learning
#   tabulate: draw table

#   https://pytorch.org/get-started/locally/ (stable(1.12.0), Linux, Conda, Python, CUDA11.6)
#   conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

# Record: 
#   Source data: 

# Execute: 
##  e1
#   nohup python B_train.py > TR/e1_test_model/pre50_0_m1.txt 2> err.txt &
#   nohup python B_train.py > TR/e1_test_model/pre50_0_m2.txt 2> err.txt &
#   nohup python B_train.py > TR/e1_test_model/pre50_0_m3.txt 2> err.txt &
#   nohup python B_train.py > TR/e1_test_model/pre50_0_m4.txt 2> err.txt &
#   nohup python B_train.py > TR/e1_test_model/pre50_0_m5.txt 2> err.txt &

#   nohup python B_train.py > TR/e1_test_model/pre50_01_m1.txt 2> err.txt &
#   nohup python B_train.py > TR/e1_test_model/pre50_01_m2.txt 2> err.txt &
#   nohup python B_train.py > TR/e1_test_model/pre50_01_m3.txt 2> err.txt &
#   nohup python B_train.py > TR/e1_test_model/pre50_01_m4.txt 2> err.txt &
#   nohup python B_train.py > TR/e1_test_model/pre50_01_m5.txt 2> err.txt &

## e2
#   nohup python B_train.py > TR/e2_test_dataset_0/preN_0_m4.txt 2> err.txt &

## e3
#   nohup python B_train.py > TR/e3_test_dataset_1/preN_1_m4.txt 2> err.txt &

## e4
#   nohup python B_train.py > TR/e4_test_dataset_01/pre100_01_m4.txt 2> err.txt &
#   nohup python B_train.py > TR/e4_test_dataset_01/pre300_01_m4.txt 2> err.txt &
#   nohup python B_train.py > TR/e4_test_dataset_01/pre666_01_m4.txt 2> err.txt &
#   nohup python B_train.py > TR/e4_test_dataset_01/pre1331_01_m4.txt 2> err.txt &
#   nohup python B_train.py > TR/e4_test_dataset_01/pre3328_01_m4.txt 2> err.txt &
#   nohup python B_train.py > TR/e4_test_dataset_01/pre6655_01_m4.txt 2> err.txt &

## e5
#   nohup python B_train.py > TR/e5_test_dataset_012/preN_012_m4.txt 2> err.txt &

## e6
#   nohup python B_train.py > TR/e6_test_dataset_012p_chi/preN_012p_chi_m4.txt 2> err.txt &

## e7
#   nohup python B_train.py > TR/e7_test_dataset_012p_eng/preN_012p_eng_m4.txt 2> err.txt &

## e8
#   nohup python B_train.py > TR/e8_test_dataset_01p_eng/preN_01p_eng_m4.txt 2> err.txt &

# Reference:
#   Multi-label classification stratify:
#       https://zhuanlan.zhihu.com/p/344329369

import pandas
import os, sys, time, re
from datetime import datetime
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

totalTime = time.time()

def tcfunc(x, n=4): ## trancate a number to have n decimal digits
    d = '0' * n
    d = int('1' + d)
    if isinstance(x, (int, float)): return int(x * d) / d
    return x

def show_Result(predictions, test_yL):
    print('MicroF1 = %0.4f, MacroF1 = %0.4f' %
        (metrics.f1_score(test_yL, predictions, average='micro'),
        metrics.f1_score(test_yL, predictions, average='macro')) + '\n')

    print('\t\tPrecision\tRecall\t\tF1\t\t\tSupport')

    (Precision, Recall, F1, Support) = list(map(tcfunc, 
        precision_recall_fscore_support(test_yL, predictions, average='micro')))
    print('Micro\t{}\t\t{}\t\t{}\t\t{}'.format(Precision, Recall, F1, Support))

    (Precision, Recall, F1, Support) = list(map(tcfunc, 
        precision_recall_fscore_support(test_yL, predictions, average='macro'))) 
    print('Macro\t{}\t\t{}\t\t{}\t\t{}\n'.format(Precision, Recall, F1, Support))

    print('Test accuracy is %1.4f'%(accuracy_score(test_yL, predictions)))
    print(classification_report(test_yL, predictions))

### Global variable
# modelCode = 'm1'; modelClass = 'bert'; modelName = 'bert-base-cased'
# modelCode = 'm2'; modelClass = 'bert'; modelName = 'bert-base-uncased'
# modelCode = 'm3'; modelClass = 'bert'; modelName = 'bert-base-multilingual-cased'
modelCode = 'm4'; modelClass = 'distilbert'; modelName = 'distilbert-base-multilingual-cased'
# modelCode = 'm5'; modelClass = 'bert'; modelName = 'hfl/chinese-macbert-large'

numOfEpochs = 10

### Read data
## Num of data: 191783
## {'Chinese': 61465, 'English': 130013, 'Else': 305}
## mix_df: 191478
## pre50_df: 9910
## pre100_df: 15015
## pre300_df: 27355

## Num of subject: 66548
## 1 %: 665.48  -> 666  (98); N: 42048 , 21.9 %
## 2 %: 1330.96 -> 1331 (55); N: 58028 , 30.3 %
## 5 %: 3327.4  -> 3328 (25); N: 85642 , 44.7 %
## 10%: 6654.8  -> 6655 (12); N: 109694, 57.2 %

## 15%: 9982.2  -> 9983 (8) ; N: 124665, 65.1 %
## 20%: 13309.6 -> 13310(5) ; N: 135168, 70.5 %
## 25%: 16637(4)            ; N: 143391, 74.8 %
## '索書號', '書名', '作者', '語言', '主題詞'

# preN = 50; dataSet = f'pre{preN}'
for preN in [50, 100, 300, 666, 1331, 3328, 6655]:
    dataSet = f'pre{preN}'
    df = pandas.read_csv(f'./source/{dataSet}_df.csv', lineterminator='\n')

    ### Create text_df
    def first_three_yards(pcn):
        match = re.search(r'(\d{3})', pcn)
        if match: return match.group(1)
        else: return '000'

    ## Text = 0
    # df0 = df.iloc[:,[0]]
    # for ind, row in df0.iterrows(): row['索書號'] = first_three_yards(row['索書號'])
    # df0.rename(columns = {'索書號':'text'}, inplace = True)
    # text_nparr = df0.to_numpy()

    ## Text = 1
    # df1 = df.iloc[:,[1]]
    # # for ind, row in df1.iterrows(): row['書名'] = first_three_yards(row['索書號'])
    # df1.rename(columns = {'書名':'text'}, inplace = True)
    # text_nparr = df1.to_numpy()

    ## Text = 0-1
    # df0 = df.iloc[:,[0]]
    # for ind, row in df0.iterrows(): row['索書號'] = first_three_yards(row['索書號'])
    # df1 = df.iloc[:,[1]]
    # text_df = df0['索書號'].map(str) + ' ' + df1['書名'].map(str)
    # text_nparr = pandas.DataFrame(data=text_df, columns=['text']).to_numpy()

    ## Text = 0-1-2
    # df0 = df.iloc[:,[0]]
    # for ind, row in df0.iterrows(): row['索書號'] = first_three_yards(row['索書號'])
    # df1 = df.iloc[:,[1]]
    # df2 = df.iloc[:,[2]]
    # text_df = df0['索書號'].map(str) + ' ' + df1['書名'].map(str) + ' ' + df2['作者'].map(str)
    # text_nparr = pandas.DataFrame(data=text_df, columns=['text']).to_numpy()

    ## Text = 0-1-2-p-chi
    # df0 = df.iloc[:,[0]]
    # for ind, row in df0.iterrows(): row['索書號'] = first_three_yards(row['索書號'])
    # df1 = df.iloc[:,[1]]
    # df2 = df.iloc[:,[2]]
    # text_df = '這本書的索書號是' + df0['索書號'].map(str) + '，書名是' + df1['書名'].map(str) + '，作者是' + df2['作者'].map(str) + '。'
    # text_nparr = pandas.DataFrame(data=text_df, columns=['text']).to_numpy()

    ## Text = 0-1-2-p-eng
    # df0 = df.iloc[:,[0]]
    # for ind, row in df0.iterrows(): row['索書號'] = first_three_yards(row['索書號'])
    # df1 = df.iloc[:,[1]]
    # df2 = df.iloc[:,[2]]
    # text_df = 'The permanent call number for this book is ' + df0['索書號'].map(str) + ', the name of the book is ' + df1['書名'].map(str) + ', and the author of the book is ' + df2['作者'].map(str) + '.'
    # text_nparr = pandas.DataFrame(data=text_df, columns=['text']).to_numpy()

    ## Text = 0-1-p-eng
    df0 = df.iloc[:,[0]]
    for ind, row in df0.iterrows(): row['索書號'] = first_three_yards(row['索書號'])
    df1 = df.iloc[:,[1]]
    text_df = 'The permanent call number for this book is ' + df0['索書號'].map(str) + ', the name of the book is ' + df1['書名'].map(str) + '.'
    text_nparr = pandas.DataFrame(data=text_df, columns=['text']).to_numpy()

    # print(text_nparr[:5]); exit()

    ## with prompt programming
    # text_df = '這本書的索書號為' + df0['Permanent Call Number'].map(str)\
    #             + '，這本書的書名為' + df1['Title'].map(str)\

    ### Create label_df
    label_df = df.iloc[:,[4]]
    label_df.rename(columns = {'主題詞':'label'}, inplace = True)
    subject_dict = {}
    subject_code = 0

    for ind, row in label_df.iterrows():
        for subject in str(row['label']).split(';'):
            if subject not in subject_dict: subject_dict[subject] = subject_code; subject_code += 1
            else: continue

    def multi_label_to_list(subjects):
        label_list = [0]*int(dataSet[3:])
        for subject in subjects.split(';'): label_list[subject_dict[subject]] = 1
        return label_list

    for ind, row in label_df.iterrows(): row['label'] = multi_label_to_list(row['label'])
    label_nparr = pandas.DataFrame([lists for lists in label_df['label'].values]).to_numpy()
    # print(label_nparr[:5]); exit()

    ### Train test split
    ## Single-label
    # x_train, x_test, y_train, y_test = train_test_split(text_df, label_df, train_size = 0.9, stratify = label_df)
    ## Multi-label
    # train_X,train_y,test_X,test_y  = iterative_train_test_split(x,y,test_size = 0.2903)
    train_text, train_label, test_text, test_label = iterative_train_test_split(text_nparr, label_nparr, test_size = 0.1)
    # print(train_text); print(train_label); print(test_text); print(test_label); exit()

    ### Create train_df
    train_text_df = pandas.DataFrame(train_text, columns =['text'])
    train_label_df = pandas.DataFrame({'label':list(train_label)}, columns =['label'])
    # print(test_text); exit()
    train_df = pandas.concat([train_text_df, train_label_df], axis = 1, ignore_index = True)
    train_df.rename(columns = {0:'text', 1:'label'}, inplace = True)
    # print(train_df.head()); exit()

    ### Create predict Q/A list
    test_text_list = [lists[0] for lists in test_text]
    # print(test_text_list[:5]); exit()
    test_label_list = test_label
    # print(test_label_list[:5]); exit()

    ### Model Definition
    model = MultiLabelClassificationModel(
        #'bert', 'bert-base-chinese', ## (model class, checkpoints=='model weights')
        f'{modelClass}', f'{modelName}', ## run at ai2 gpu1
        num_labels = int(dataSet[3:]), ## 0~29
        args = {## learning rate(學習率) ## 更新幅度
            'reprocess_input_data': True, 
            'overwrite_output_dir': True, 
            'num_train_epochs': numOfEpochs, 
            'max_seq_length': 128, ## max = 512 ## Avg Text Chars =  76
            'train_batch_size': 64, ## 一次送幾筆
            'eval_batch_size': 64, ## 幾筆預測一次
            'output_dir': f'outputs/model_{modelCode}'
        }, 
        use_cuda = True, 
        cuda_device = 1, ## 0: first GPU, 1: second GPU
    )
    print(f'preN = {preN}')
    print(f'modelName = {modelName}\n')

    ### Train the model
    trainTime = time.time()
    model.train_model(train_df)
    print('training time: %1.2f'%(time.time()-trainTime) + '\n')

    ### Make predictions with the model
    predictions, raw_outputs = model.predict(test_text_list)
    show_Result(predictions, test_label_list) ## (Q, A)
print('Total time: %1.2f'%(time.time()-totalTime) + '\n')