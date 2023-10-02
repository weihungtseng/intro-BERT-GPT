# Topic: NTNU 1-1 Final - Statistic
# Tag: [Statistic] [NTNU Library Data]
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
#   python A_statistic.py
#   nohup python A_statistic.py &

# Reference:
#   Stratified K Fold Cross Validation:
#       https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/
#       https://stats.stackexchange.com/questions/555222/converting-a-code-for-5-fold-cross-validation-to-stratified-5-fold-cross-validat

import os, sys, time, re
import pandas
from tqdm import tqdm
from tabulate import tabulate

totaltime = time.time()
### Create target column df
## Read data from excel(file, worksheet, col name, col index)
## Num of data: 191783
# source_path = './source/iso-c修正過書目_final.xlsx'
# df = pandas.read_excel(
#     source_path,
#     'Sheet1', ## Work sheet
#     names = ['作者', '索書號', '語言', '書名', '主題詞'],
#     usecols = 'O:S'
# )
# df = pandas.concat(
#     [
#         df.iloc[:, [1]], 
#         df.iloc[:, [3]], 
#         df.iloc[:, [0]], 
#         df.iloc[:, [2]], 
#         df.iloc[:, [4]], 
#     ], 
#     axis = 1, 
#     ignore_index = True
# )
# df.rename(columns = {0:'索書號', 1:'書名', 2:'作者', 3:'語言', 4:'主題詞'}, inplace = True)
# df.to_csv('./source/target_df.csv', index=False)
# print('\n\n' + 'Total time: %1.2f'%(time.time()-totaltime) + '\n')
# exit()

### Statistic chi/eng/else
# df = pandas.read_csv(f'./source/target_df.csv', lineterminator='\n')
# language_dict = {'Chinese':0, 'English':0, 'Else':0}
# for ind, row in df.iterrows():
#     if row['語言'] == 'Chinese': language_dict['Chinese'] += 1
#     elif row['語言'] == 'English': language_dict['English'] += 1
#     else: language_dict['Else'] += 1
# print(language_dict)
# ## {'Chinese': 61465, 'English': 130013, 'Else': 305}
# exit()

### Create mix/chi/eng df & filter 'Else' data
# mix_df = pandas.DataFrame(
#     columns = [
#         '索書號', 
#         '書名', 
#         '作者', 
#         '語言', 
#         '主題詞'
#     ]
# )

# chi_df = pandas.DataFrame(
#     columns = [
#         '索書號', 
#         '書名', 
#         '作者', 
#         '語言', 
#         '主題詞'
#     ]
# )

# eng_df = pandas.DataFrame(
#     columns = [
#         '索書號', 
#         '書名', 
#         '作者', 
#         '語言', 
#         '主題詞'
#     ]
# )

# df = pandas.read_csv(f'./source/target_df.csv', lineterminator='\n')
# progress = tqdm(total=len(df))
# for ind, row in df.iterrows():
#     if row['語言'] == 'Chinese':
#         chi_df.loc[len(chi_df)] = [row['索書號'], row['書名'], row['作者'], row['語言'], row['主題詞']]
#     elif row['語言'] == 'English':
#         eng_df.loc[len(eng_df)] = [row['索書號'], row['書名'], row['作者'], row['語言'], row['主題詞']]
#     else:
#         continue
#     mix_df.loc[len(mix_df)] = [row['索書號'], row['書名'], row['作者'], row['語言'], row['主題詞']]
#     progress.update(1)
# mix_df.to_csv('./source/mix_df.csv', index=False)
# chi_df.to_csv('./source/chi_df.csv', index=False)
# eng_df.to_csv('./source/eng_df.csv', index=False)
# print('\n\n' + 'Total time: %1.2f'%(time.time()-totaltime) + '\n')
# ## Total time: 2136.33
# exit()

### Statistic num of data for each Subject
# subject_dict = {}

# df = pandas.read_csv(f'./source/mix_df.csv', lineterminator='\n')
# # df = pandas.read_csv(f'./source/chi_df.csv', lineterminator='\n')
# # df = pandas.read_csv(f'./source/eng_df.csv', lineterminator='\n')
# for ind, row in df.iterrows():
#     for subject in row['主題詞'].split(';'):
#         if subject in subject_dict:
#             subject_dict[subject] += 1
#         else:
#             subject_dict[subject] = 1
# subject_dict = dict(sorted(subject_dict.items(), key=lambda item: item[1], reverse=True))

# print('主題詞\t\t數量')
# for key, val in subject_dict.items():
#     print(f'{val}\t\t{key}')

# print(f'mix_df num of subject = {len(subject_dict)}')
# # print(f'chi_df num of subject = {len(subject_dict)}')
# # print(f'eng_df num of subject = {len(subject_dict)}')

# ## mix_df num of subject = 66548
# ## chi_df num of subject = 14481
# ## eng_df num of subject = 56516
# exit()

### Filter label
## PreN
# N = 50; boundery = '中國文學'
# N = 100; boundery = 'History'
# N = 300; boundery = 'Economics.'

## 1 %: 665.48  -> 666(98 Mayas)
# N = 666; boundery = 'Mayas'
## 2 %: 1330.96 -> 1331(55 Conduct of life)
# N = 1331; boundery = 'Conduct of life'
## 5 %: 3327.4  -> 3328(25 Developmental psychobiology.)
# N = 3328; boundery = 'Developmental psychobiology.'
## 10%: 6654.8  -> 6655(12 Neoclassicism (Art))
# N = 6655; boundery = 'Neoclassicism (Art)'
## 15%: 9982.2  -> 9983(8 African American soldiers)
# N = 9983; boundery = 'African American soldiers'
## 20%: 13309.6 -> 13310(5 Medical jurisprudence)
# N = 13310; boundery = 'Medical jurisprudence'
## 25%: 16637(4 書影)
# N = 16637; boundery = '書影'

subject_dict = {}
preN_list = []
test_dict = {}

preN_df = pandas.DataFrame(
    columns = [
        '索書號', 
        '書名', 
        '作者', 
        '語言', 
        '主題詞'
    ]
)

df = pandas.read_csv(f'./source/mix_df.csv', lineterminator='\n')
for ind, row in df.iterrows():
    for subject in row['主題詞'].split(';'):
        if subject in subject_dict:
            subject_dict[subject] += 1
        else:
            subject_dict[subject] = 1
subject_dict = dict(sorted(subject_dict.items(), key=lambda item: item[1], reverse=True))
for key, val in subject_dict.items():
    preN_list.append(key); # test_dict[key] = val
    if key == boundery: break

# print('數量\t\t主題詞')
# for key, val in test_dict.items():
#     print(f'{val}\t\t{key}')
# exit()

for ind, row in df.iterrows():
    legal = True
    for subject in row['主題詞'].split(';'):
        if subject not in preN_list: legal = False; break
    if legal: preN_df.loc[len(preN_df)] = [row['索書號'], row['書名'], row['作者'], row['語言'], row['主題詞']]

preN_df.to_csv(f'./source/pre{N}_df.csv', index=False)
print('\n\n' + 'Total time: %1.2f'%(time.time()-totaltime) + '\n')
exit()