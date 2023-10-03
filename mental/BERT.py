### Install:
'''
    pip install transformers simpletransformers emoji==0.6.0 openpyxl torch

    https://pytorch.org/get-started/locally/ (stable(1.12.0), Linux, Conda, Python, CUDA11.6)
    For ai2:
        conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    For ai3
        conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c conda-forge
'''

### Record: 
'''

'''

### Execute:
'''
    python A_k_fold_cross_validation_v2.py
    nohup python A_k_fold_cross_validation_v2.py &
'''

### Reference:
'''
    Stratified K Fold Cross Validation:
        https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/
        https://stats.stackexchange.com/questions/555222/converting-a-code-for-5-fold-cross-validation-to-stratified-5-fold-cross-validat

    Python | Merge two lists into list of tuples:
        https://www.geeksforgeeks.org/python-merge-two-lists-into-list-of-tuples/

'''

import os, sys, time, re
import pandas
from datetime import datetime
from tqdm import tqdm
import gc
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

### Global variable
sTotaltime = time.time()
today = datetime.today().strftime('%Y-%m-%d')
features = ["M", "R", "S", "HA", "HE"]
data = ['1017-M面向檢核後訓練資料', '1017-R面向檢核後訓練資料', '1017-S面向檢核後訓練資料', '0313-HA面向訓練資料更新', '1017-HE面向檢核後訓練資料']
cols = ["M1", "M5", "M10", "R1", "R2", "S1", "S2", "HA1", "HA2", "HE1", "HE2"]
model_template = 'vinai/bertweet-base'
k_fold = 5
resultFilePath = f"./A_result/formal/20230313_fold/{today}_{k_fold}_fold_R2_HE2.txt"

def plus2(x):
    if x >= -2 and x <= 2: return int(x+2)
    return 0

def tcfunc(x, n=4): ## trancate a number to have n decimal digits
    d = '0' * n
    d = int('1' + d)
    if isinstance(x, (int, float)): return int(x * d) / d
    return x

def show_Result(predictions, test_yL):
    f = open(resultFilePath, "a")
    f.write('MicroF1 = %0.4f, MacroF1 = %0.4f\n\n' %
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
del lst; del M_df; del R_df; del S_df; del HA_df; del HE_df; del zero_df
print(df.head()); print()

for col in cols:
    df1 = df[['tweet', col]]
    df1.rename(columns = {'tweet':'text', f'{col}':'labels'}, inplace = True)

    ## k-fold cross validation
    skf = StratifiedKFold(n_splits = k_fold, shuffle = True, random_state = 1)
    # for train_index, test_index in skf.split(df1['text'], df1['labels']):
    for j, train_test in enumerate(skf.split(df1['text'], df1['labels'])):
        train_index, test_index = train_test
        text_train_fold, text_test_fold = df1['text'][train_index], df1['text'][test_index]
        labels_train_fold, labels_test_fold = df1['labels'][train_index], df1['labels'][test_index]
        train_df = pandas.concat([text_train_fold, labels_train_fold], axis=1, join='inner')

        ## Create a MultiLabelClassificationModel-5
        model = ClassificationModel(
            'bertweet', f'{model_template}', ## (model class, model)
            num_labels = 5, ## 0, 1, 2, 3, 4
            args = {
                'reprocess_input_data': True, 
                'overwrite_output_dir': True, 
                # 'use_multiprocessing': False, 
                # 'horizontal_flip': False, 
                'num_train_epochs': 10, 
                'max_seq_length': 256, ## max = 512
                'train_batch_size': 64, ## 一次送幾筆
                'eval_batch_size': 64, ## 幾筆預測一次
                'output_dir': f'outputs/v2/{k_fold}_foldCV_{col}_{today}'
            }, 
            use_cuda = True, 
            # use_cuda = False, 
            cuda_device = 2, ## 0: first GPU, 1: second GPU
            # ignore_mismatched_sizes=True
        )

        stime = time.time()
        ## Train the model
        model.train_model(train_df)
        f = open(resultFilePath, "a")
        f.write(f"model = {model_template}\n")
        f.write("training time: %1.2f\n"%(time.time()-stime))
        f.write(f"col = {col}\n")
        f.write(f"{k_fold}-fold cross validation = {j+1}\n\n")
        f.close()

        ## Make predictions with the model
        predictions, raw_outputs = model.predict(list(text_test_fold))
        num = len(labels_test_fold)
        show_Result(predictions[0:num], list(labels_test_fold))

f = open(resultFilePath, "a")
f.write("\n\n" + "Total time: %1.2f"%(time.time()-sTotaltime) + "\n")
f.close()

## 20230313_fold total time: 127666.62
## [1] 3057344