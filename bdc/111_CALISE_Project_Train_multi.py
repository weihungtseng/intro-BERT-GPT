# Topic: 2022 Big Data Competition
# Tag: [Library Data] [Predict Subject]
# Author: Wei-Hung, Tseng
# CreateDate: 2022/07/07
# SSH: 
# Conda env: 2022BDC
# Install: 
#   pip install transformers simpletransformers openpyxl datetime arrow

#   https://pytorch.org/get-started/locally/ (stable(1.12.0), Linux, Conda, Python, CUDA11.6)
#   conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

# Complie: python -m py_compile 111_CALISE_Project_Train_multi.py

# Execute: 
#   2022/07/07
#   python 111_CALISE_Project_Train.py > train_result.txt
#   nohup python 111_CALISE_Project_Train.py > TR/bert_base_chinese_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/bert_base_chinese_100000.txt 2> err.txt &
#   2022/07/08
#   nohup python 111_CALISE_Project_Train.py > TR/b_b_multilingual_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/chi_macbert_large_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/b_b_multilingual_296967.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/b_b_cased_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/b_b_uncased_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/distilbert_multilingual_cased_10000.txt 2> err.txt &
#   2022/07/10
#   nohup python 111_CALISE_Project_Train.py > TR/b_b_multilingual_296967.txt 2> err.txt &
#   2022/07/11
#   python 111_CALISE_Project_Train.py > TR/b_b_multilingual_10000.txt 2> err.txt
#   nohup python 111_CALISE_Project_Train.py > TR/b_b_multilingual_50000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/b_b_multilingual_100000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/b_b_multilingual_150000.txt 2> err.txt &
#   2022/07/11
#   nohup python 111_CALISE_Project_Train.py > TR/b_b_multilingual_200000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/b_b_multilingual_250000.txt 2> err.txt &
#   2022/07/14
#   nohup python 111_CALISE_Project_Train.py > TR/nlp_b_b_multilingual_10000.txt 2> err.txt & (failed)
#   nohup python 111_CALISE_Project_Train.py > TR/4_6_10_bb_multilingual_10000.txt 2> err.txt & (failed)
#   2022/07/15
#   nohup python 111_CALISE_Project_Train.py > TR/4_6_10_bb_multilingual_10000.txt 2> err.txt &
#   2022/07/18
#   nohup python 111_CALISE_Project_Train.py > TR/4_6_10_bb_multilingual_50000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/4_6_10_bb_multilingual_100000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/4_6_10_bb_multilingual_150000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train.py > TR/4_6_10_bb_multilingual_200000.txt 2> err.txt &
#   2022/07/21
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_100.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_211028.txt 2> err.txt &
#   2022/07/22
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_50000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_100000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_150000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_200000.txt 2> err.txt &
#   2022/07/26
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e20_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e30_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e40_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e50_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e10c1_10000.txt 2> err.txt &
#   2022/07/27
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e10c2_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e10c3_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e10c4_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e10c5_10000.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e10c6_10000.txt 2> err.txt &
#   2022/07/29
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e10c7_10000.txt 2> err.txt &
#   2022/08/01
#   nohup python 111_CALISE_Project_Train_multi.py > TR/multi_4_6_10_bb_multilingual_e10c8_10000.txt 2> err.txt &
#   2022/08/23
#   nohup python 111_CALISE_Project_Train_multi.py > TR/train_ds05_e10.txt 2> err.txt &
#   2022/08/24
#   nohup python 111_CALISE_Project_Train_multi.py > TR/train_ds06_e10.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/train_ds07_e10.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/train_ds08_e10.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/train_ds09_e10.txt 2> err.txt &
#   2022/08/25
#   nohup python 111_CALISE_Project_Train_multi.py > TR/train_ds098_e10.txt 2> err.txt &
#   2022/09/02
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_bb_chinese_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_distilbert_multilingual_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_bert_multilingual_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_chi_macbert_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_bb_cased_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_bb_uncased_1w.txt 2> err.txt &

#   nohup python 111_CALISE_Project_Train_multi.py > TR/BM_merge_train.txt 2> err.txt &

#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_bb_chinese_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_distilbert_multilingual_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_bert_multilingual_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_chi_macbert_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_bb_cased_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_bb_uncased_1w.txt 2> err.txt &

#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10_model_1w/4_6_10_bb_chinese_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10_model_1w/4_6_10_distilbert_multilingual_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10_model_1w/4_6_10_bert_multilingual_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10_model_1w/4_6_10_chi_macbert_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10_model_1w/4_6_10_bb_cased_1w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10_model_1w/4_6_10_bb_uncased_1w.txt 2> err.txt &

#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_d_ds/4_6_10p_d_ds05_e10.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_d_ds/4_6_10p_d_ds06_e10.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_d_ds/4_6_10p_d_ds07_e10.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_d_ds/4_6_10p_d_ds08_e10.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_d_ds/4_6_10p_d_ds09_e10.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_d_ds/4_6_10p_d_ds098_e10.txt 2> err.txt &

#   nohup python 111_CALISE_Project_Train_multi.py > TR/1w_10w/4_d_ds07_e10_2468w.txt 2> err.txt &
#   nohup python 111_CALISE_Project_Train_multi.py > TR/1w_10w/4_6_10p_d_ds06_e10_2468w.txt 2> err.txt &

#   nohup python 111_CALISE_Project_Train_multi.py > TR/2w_5w/4_6_10p_d_ds06_e10_wq.txt 2> err.txt &

#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_BM_eX_3w/BM_eX_3w.txt 2> err.txt &

#   nohup python 111_CALISE_Project_Train_multi.py > TR/4_6_10p_BM_e10_3w/BM_e10_3w.txt 2> err.txt &

# Reference:
#   Save model checkpoint every 3 epochsPermalink:
#       https://simpletransformers.ai/docs/tips-and-tricks/#save-model-checkpoint-every-3-epochs

# Record: 'bert', 'xlnet', 'xlm', 'roberta', 'distilbert'
#   c1: 'learning_rate': 1e-6
#   c2: 'learning_rate': 2e-5
#   c3: 'learning_rate': 4e-5
#   c4: 'learning_rate': 6e-5
#   c5: 'learning_rate': 8e-5
#   c6: 'learning_rate': 1e-4
#   c7: 'learning_rate': 2e-4
#   c8: 'learning_rate': 4e-4

## Import Library & Global Variable
import pandas
import numpy as np
import sys, time, arrow
from datetime import datetime
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pprint import pprint

## Get execute datetime
# today = arrow.get(str(datetime.today())).shift(hours=8)
# print(f"Execute time: {str(today).replace('T', ' ').replace('+00:00', '')}")
print(f"Execute time: {datetime.today()}")

## Global variable
label_type = 'multi'
## All train data = 211028
# numOfData = 10000
## Train data after delete by similarity:
##      05: 112803|06: 155819|07: 183762|08: 198411|09: 204830|098: 206121
# Feature = '4_6_10'
Feature = '4_6_10p'
# modelName = 'bert-base-chinese'
modelName = 'bert-base-multilingual-cased'
# modelName = 'distilbert-base-multilingual-cased' ## distilbert
# modelName = 'hfl/chinese-macbert-large' ## bert
# modelName = 'bert-base-cased'
# modelName = 'bert-base-uncased'
# numOfepochs = 10
numOfLearningRate = 4e-4
# fileOfTrain = 'merge_train_dataset'
# fileOfTrain = '4_DL_similar07'
fileOfTrain = '4_6_10p_DL_similar06'

## trancate a number to have n decimal digits
def tcfunc(x, n=4):
    d = '0' * n
    d = int('1' + d)
    if isinstance(x, (int, float)): return int(x * d) / d
    return x

def print_cls_report(prediction, y_true):
    print('Test accuracy is %1.4f'%(accuracy_score(y_true, prediction)))
    print(classification_report(y_true, prediction))
    
    print("\tPrecision\tRecall\tF1\tSupport")
    (Precision, Recall, F1, Support) = list(map(tcfunc, 
        precision_recall_fscore_support(y_true, prediction, average='micro')))
    print("Micro\t{}\t{}\t{}\t{}".format(Precision, Recall, F1, Support))

    (Precision, Recall, F1, Support) = list(map(tcfunc, 
        precision_recall_fscore_support(y_true, prediction, average='macro')))
    print("Macro\t{}\t{}\t{}\t{}".format(Precision, Recall, F1, Support))
    
    if False:
        print(confusion_matrix(y_true, prediction))
        try: 
            print(classification_report(y_true, prediction, digits=4))
        except ValueError:
            print('May be some category has no predicted samples')
        show_confusion_matrix(prediction)

    print(f'y_true.shape={y_true.shape}, prediction.shape={prediction.shape}')
    
    pred = prediction
    if not isinstance(pred, np.ndarray): pred = prediction.toarray()
    print(type(y_true), type(prediction), type(pred))
    try:
        print('macro roc_auc_score is %1.4f'%(roc_auc_score(y_true, pred, average='macro'))) # default average=’macro’
        print('micro roc_auc_score is %1.4f'%(roc_auc_score(y_true, pred, average='micro')))
    except:
        print("roc_auc_score error!!!")
    try:
        fpr, tpr, thresholds = roc_curve(y_true, pred)
        print(f'fpr={fpr}\ntpr={tpr}\nthresholds={thresholds}')
    except:
        print('roc_curve error!!!')

def clean4Text(text):
        if text[-1] == '/': text = text[:-1]
        return text

#### Main
## 4:  05: 112803|06: 155819|07: 183762|08: 198411|09: 204830|098: 206121
## 4_6_10p  05: 166087|06: 189022|07: 200067|08: 205420|09: 207866|098: 208833
# for numOfData in [10000, 50000, 100000, 166087]:
# for numOfData in [10000, 100000, 211028]:
# for numOfData in [25000, 30000, 35000, 45000]:
for numOfepochs in [10]:
    numOfData = 30000
    print(f"Train data file = {fileOfTrain}")
    ## Read train data(14 column)
    df = pandas.read_csv(f"../train_data/{fileOfTrain}.csv", 
                        lineterminator='\n')
    df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
    df.rename(columns={'Subjects\r': 'Subjects'}, inplace=True)

    ## Create multi-label text_df
    ## 4
    # df1 = df.iloc[:,[4]]
    # df1['text'] = df1['Title'].apply(lambda x: clean4Text(x))
    # text_df = df1.iloc[:,[1]]

    ## 4-6-10
    # df1 = df.iloc[:,[4]]
    # df2 = df.iloc[:,[6]]
    # df3 = df.iloc[:,[10]]
    # text_df = df1['Title'].map(str) + df2['Author'].map(str) + df3['Publisher'].map(str)

    ## 4-6-10 and prompt programming
    df1 = df.iloc[:,[4]]
    df2 = df.iloc[:,[6]]
    df3 = df.iloc[:,[10]]
    text_df = df1['Title'].map(str)\
            + '這本書的作者是' + df2['Author'].map(str)\
            + '而它的出版社是' + df3['Publisher'].map(str)

    ## Create multi-label subject_df
    subject_dict = {'United States': 0, 'History': 1, '中國': 2, 'Politics and government': 3, 'Great Britain': 4, 'Congresses': 5, 'Philosophy': 6, 'China': 7, '傳記': 8, 'History and criticism': 9, '歷史': 10, 'Social aspects': 11, 'Biography': 12, 'Foreign relations': 13, '臺灣': 14, 'Europe': 15, 'Economic conditions': 16, 'Psychology': 17, '論文': 18, 'Social conditions': 19, 'Management': 20, 'Case studies': 21, 'Education': 22, 'Economic policy': 23, 'Law and legislation': 24, 'Social sciences': 25, 'Research': 26, 'Finance': 27, '哲學': 28, '文集': 29}
    subject_df = df.iloc[:,[13]]

    def multi_label(subjects):
        label_list = [0]*30
        for subject in subjects.split('; '):
            if subject in subject_dict: label_list[subject_dict[subject]] = 1
        return label_list

    subject_df['labels'] = subject_df['Subjects'].apply(lambda x: multi_label(x))
    subject_df = subject_df.iloc[:,[1]]

    ## Create train_df
    train_df = pandas.concat([text_df, subject_df], axis=1, join='inner')
    train_df.rename(columns = {0:'text'}, inplace = True)
    train_df = train_df.sample(frac=1)[0:numOfData]

    print(f"label_type = '{label_type}'")
    print(f"Feature = '{Feature}'")
    print(f"Number of train data = {numOfData}")
    print(f"Model = '{modelName}'\n")

    myargs = {
        'reprocess_input_data': True, 
        'overwrite_output_dir': True, 
        'num_train_epochs': numOfepochs, 
        'max_seq_length': 128, 
        'train_batch_size': 64, 
        'eval_batch_size': 64, 
        'save_model_every_epoch': True, 
        'output_dir': f'outputs/model_{label_type}_{Feature}_{modelName}_e{numOfepochs}_{numOfData}'
        # 'output_dir': f'outputs/2022BDC_best_model'
    }
    pprint(myargs)

    ## Modle Definition
    model = MultiLabelClassificationModel(
        #'bert', 'bert-base-chinese', ## (model class, checkpoints=="model weights")
        'bert', f'{modelName}', 
        num_labels = 30, ## 0~29
        args = {## learning rate(學習率) ## 更新幅度
            'reprocess_input_data': True, 
            'overwrite_output_dir': True, 
            'num_train_epochs': numOfepochs, 
            'max_seq_length': 128, ## max = 512 ## Avg Text Chars =  76
            'train_batch_size': 64, 
            'eval_batch_size': 64, 
            'save_model_every_epoch': True, 
            # 'output_dir': f'outputs/model_{label_type}_{Feature}_{modelName}_e{numOfepochs}_{numOfData}'
            'output_dir': 'outputs/2022BDC_best_model'
        }, 
        use_cuda = True, 
        # cuda_device = 1, ## 0: first GPU, 1: second GPU
    )
    print("\nModel setting success")

    ## Training ....
    print("\nStart train......")
    stime = time.time()
    ## Train the model
    model.train_model(train_df)
    print("training time: %1.2f"%(time.time()-stime) + "\n")

    ## Read test data(14 column)
    time2 = time.time()
    df = pandas.read_csv("../test_data/merge_1w_test_dataset_for_user.csv", 
                        lineterminator='\n')
    df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
    df.rename(columns={'Subjects\r': 'Subjects'}, inplace=True)

    ## Create multi-label text_df
    df1 = df.iloc[:,[4]]
    df2 = df.iloc[:,[6]]
    df3 = df.iloc[:,[10]]
    text_df = df1['Title'].map(str)\
            + '這本書的作者是' + df2['Author'].map(str)\
            + '而它的出版社是' + df3['Publisher'].map(str)
    text_df = pandas.DataFrame({'text': text_df})

    ## Create multi-label subject_df
    subject_df = df.iloc[:,[13]]
    subject_df['labels'] = subject_df['Subjects'].apply(lambda x: multi_label(x))
    subject_df = subject_df.iloc[:,[1]]

    ## Create test_df
    test_df = pandas.concat([text_df, subject_df], axis=1, join='inner')
    test_df.rename(columns = {0:'text'}, inplace = True)

    ## Create test_text_list & test_labels_list
    test_text_list = text_df['text'].tolist()
    test_labels_list = subject_df['labels'].tolist()

    ## Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_df)

    type(model_outputs)

    threshold = lambda x: 1 if x>0.5 else 0 
    predictions = np.array([[threshold(x) for x in row] for row in model_outputs])
    print(type(predictions), type(test_df['labels']))
    print('predictions:')
    print(predictions)

    print_cls_report(predictions, np.array(test_labels_list)) ## (Q, A)
    print("Prediction time: %1.2f"%(time.time()-time2) + "\n\n\n")

print("\n\nfinish")