#코드작성자: 성균관대학교 19학번 최새미
# -----------------------------------------------------------------------------------------------------------------------------------------------
import argparse
from collections import OrderedDict
from itertools import *
from os import listdir
import os
import pandas as pd
import numpy as np
import unicodedata
import random
import re
import keras

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import TFBertModel, TFBertMainLayer, BertConfig, BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.layers import Dense, Conv1D, Concatenate, Flatten, Embedding
from tensorflow.keras import Model, metrics
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score,classification_report

import transformers
from tqdm import tqdm
# -----------------------------------------------------------------------------------------------------------------------------------------------

# 데이터 로딩
def load_data(path, mode, label_name='LABEL 1'):

  if mode == 'train':
      columns = ['내용', label_name]
  elif mode == 'predict':
      columns = ['내용']

  print("#### 약관 데이터 loading\n")
  if os.path.isdir(path):
    contents = pd.DataFrame([], columns=columns)
    for file in listdir(path):
      print(file)
      content = pd.read_csv(os.path.join(path, file), encoding='utf-8-sig')[columns]
      contents = pd.concat([contents, content]).reset_index(drop=True)
    return contents
  elif os.path.isfile(path):
    content = pd.read_csv(path, encoding='utf-8-sig')
    return content


# 공장/불공정 조항 비율 확인
def check_unfair_rate(contents, label_name='LABEL 1'):
  print('\n\n#### 공정/불공정 조항 비율 확인\n')
  v = contents[label_name].value_counts()
  print(v)
  print('\n',round(v[1]/(v[0]+v[1]), 5),'\n\n') 


# unique한 조항만 leave
def del_duplicate(contents, mode, label_name='LABEL 1'):

  if mode == 'train':
      columns = ['내용', label_name]
  elif mode == 'predict':
      columns = ['내용']

  unique_contents = pd.DataFrame([], columns=columns)
  for con in contents['내용'].unique():
    con_df = contents[contents['내용']==con]
    unique_contents = pd.concat([unique_contents, pd.DataFrame(con_df.iloc[0]).T], axis=0, ignore_index=True)

  return unique_contents


# 글머리 기호 제거
def del_symbol(content, sub=''):
  content = re.sub(re.compile('①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑪|⑫|⑬|⑭|⑮|⑯|⑰'),sub,content)
  content = re.sub(re.compile('\d+\. *'),sub,content)
  content = re.sub(re.compile('\d+\)'),sub,content)
  content = re.sub(re.compile('\(\d+\)'),sub,content)
  content = re.sub(re.compile(' [가-하]\.'),sub,content)
  content = re.sub(re.compile('-'),sub,content)
  content = re.sub(re.compile('제\d+조 *\(.+\)'),sub,content)
  content = re.sub('\ufeff','',content)
  content = re.sub('\u200b','',content)
  content = re.sub(r'[^가-힣?.!,¿ ]','',content)
  return content.strip()

# 글머리 기호 제거
def del_symbols(contents):
  drop_idx=[]
  for i in tqdm(range(len(contents))):
    contents.iloc[i, 0] = del_symbol(contents.iloc[i, 0])
    if contents.iloc[i,0] == '':
      drop_idx.append(i)
  contents.drop(drop_idx, inplace=True)
  contents.reset_index(drop=True, inplace=True)
  return contents

# 총 전처리 
def prepare(PATH, mode, label_name='LABEL 1'):
  # 전처리
  contents = load_data(PATH, mode, label_name)
  contents = del_duplicate(contents, mode, label_name)
  contents = del_symbols(contents)
  sentences=contents['내용']
  
  # 벡터화
  input_ids=[]
  attention_masks=[]
  tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
  for sent in sentences:
      bert_inp=tokenizer.encode_plus(sent, add_special_tokens = True, max_length=128, pad_to_max_length=True, return_attention_mask=True)
      input_ids.append(bert_inp['input_ids'])
      attention_masks.append(bert_inp['attention_mask'])

  input_ids=np.asarray(input_ids)
  attention_masks=np.array(attention_masks)
  
  if mode == 'train':
    label = contents[label_name].to_numpy(dtype=int)
    return input_ids, attention_masks, label
  elif mode =='predict':
    return input_ids, attention_masks

# 모델 load & fit
def modeling(train_inp, val_inp, train_label, val_label, train_mask, val_mask, PATH, log_dir):
  bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",num_labels=2)
  model_save_path=PATH

  callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True),keras.callbacks.TensorBoard(log_dir=log_dir)]

  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

  bert_model.compile(loss=loss,optimizer=optimizer,metrics=[metric])

  print('\nBert Model',bert_model.summary())
  
  history=bert_model.fit([train_inp.astype(np.int),train_mask.astype(np.int)],train_label.astype(np.int),batch_size=64,epochs=20,validation_data=([val_inp.astype(np.int),val_mask.astype(np.int)],val_label.astype(np.int)),callbacks=callbacks)

  return bert_model


# 모델 평가 
def estimated(est_inp, est_mask, est_label, log_dir, model_save_path):
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

  trained_model = TFBertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",num_labels=2)
  trained_model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
  trained_model.load_weights(model_save_path) # 경로에 저장되어 있는 가중치 load

  preds = trained_model.predict([est_inp,est_mask],batch_size=32)
  pred_label = np.argmax(preds.logits, axis=1)
  f1 = f1_score(est_label.astype(np.int64),pred_label)
  print('F1 score',f1)
  print('Classification Report')
  print(classification_report(est_label.astype(np.int64),pred_label,target_names=['fair', 'unfair']))

  print('Training and saving built model.....')

  return trained_model


# train or predict
def run(data_path, model_path, mode, pred_path, oversample, label_name):
    if mode == 'train':
        input_ids, attention_masks, labels = prepare(data_path, mode, label_name)    
        train_inp,val_inp,train_label,val_label,train_mask,val_mask=train_test_split(input_ids, labels, attention_masks,test_size=0.2)

        true_idxs = []
        for i in range(len(train_label)):
            if int(train_label[i]) == 1:
                true_idxs.append(i)

        true_inp = train_inp[true_idxs]
        true_mask = train_mask[true_idxs]
        
        oversampled_inp = train_inp.copy()
        oversampled_mask = train_mask.copy()
        oversampled_label = train_label.copy()

        for i in range(oversample):
            oversampled_inp = np.concatenate((oversampled_inp, true_inp), axis=0)
            oversampled_mask = np.concatenate((oversampled_mask, true_mask), axis=0)
            oversampled_label = np.concatenate((oversampled_label, np.array([1]*len(true_inp)).astype(object)), axis=0)

        modeling(oversampled_inp,val_inp, oversampled_label, val_label, oversampled_mask, val_mask, model_path , log_dir='over_bert')
        estimated(val_inp, val_mask, val_label, 'over_bert', model_path)
        
    elif mode == 'predict':    
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

        model = TFBertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",num_labels=2)
        model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
        model.load_weights(model_path) # 경로에 저장되어 있는 가중치 load

        if os.isdir(data_path):
            camp_list = os.listdir(data_path)
            os.makedirs(pred_path, exist_ok=True)
            for camp in tqdm(camp_list):
                print('--------------------------------------------------------------------------------')
                print(camp)
                path = os.path.join(data_path, camp)
                contents = del_duplicate(load_data(path))
                test_inp, test_mask = prepare(path, mode, label_name)
                preds = model.predict([test_inp,test_mask],batch_size=32)
                pred_labels = np.argmax(preds.logits, axis=1)
                try:
                    contents['label'] = pred_labels
                    contents.to_csv(os.path.join(pred_path, camp),index=False, encoding='utf-8-sig')
                except:
                    print('unmatched')
                print(np.unique(pred_labels, return_counts=True))


# 메인 함수
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Input dataset path', dest='dataset', default='./label/')
    parser.add_argument('-m', '--model_path', help='Trained model path', dest='model_path')
    parser.add_argument('-k', '--mode', help='train or predict', dest='mode', default='train')
    parser.add_argument('-r', '--pred_path', help='Predicted result path', dest='pred_path', default='./predict/')
    parser.add_argument('-o', '--oversampe', help='Over Samplt rate', dest='oversample', default=1)
    parser.add_argument('-l', '--label_name', help='Label Columns Name', dest='label_name', default=1)

    args = parser.parse_args()
    
    data_path = args.dataset 
    model_path = args.model_path
    mode = args.mode
    pred_path = args.pred_path
    oversample = args.oversample
    label_name = args.label_name

    run(data_path, model_path, mode, pred_path, oversample, label_name)
# -----------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()