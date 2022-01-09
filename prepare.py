import pandas as pd
import numpy as np
import os
from os.path import join
from os import listdir
import re
import argparse

def chk_char(df,i):
    a_word = re.compile('①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑪|⑫|⑬|⑭|⑮|⑯|⑰')
    b_word = re.compile('\d+\. *')
    c_word = re.compile('\d+\)')
    d_word = re.compile('\(\d+\)')
    e_word = re.compile('[가-하]+\.')
    f_word = re.compile('-')

    if re.match(a_word, df.iloc[i, 2]):
        return 'a'
    elif re.match(b_word, df.iloc[i, 2]):
        return 'b'
    elif re.match(c_word, df.iloc[i, 2]):
        return 'c'
    elif re.match(d_word, df.iloc[i, 2]):
        return 'd'
    elif re.match(e_word, df.iloc[i, 2]):
        return 'e'
    elif re.match(f_word, df.iloc[i, 2]):
        return 'f'
    else:
        return 'g'

def seperate(path_txt, path_csv):
    txt_list = listdir(path_txt)
    name_list = [file_name.split('.')[0] for file_name in txt_list]

    for txt, name in zip(txt_list, name_list):
        with open(join(path_txt, txt), 'rb') as f:
            raw = f.read().decode("utf-8-sig")
        raw_sent = raw.split('\n')
        raw_sent = [re.sub('\r', '', sent.strip()) for sent in raw_sent]
        raw_sent = [i for i in raw_sent if i != '']

        df = pd.DataFrame(raw_sent, columns=['내용'])
        df['장'] = ''
        df['조'] = ''
        jang = False
        cho = False
        for i in range(len(df)):
            jang_word = re.compile('<* *제 *\d+ *장')
            if re.match(jang_word, df.iloc[i,0]): # 만약 내용에 장이 포함되어 있으면,
                jang = df.iloc[i,0]
            if jang: # 만약 내용에 장이 포함되어 있으면,
                df.iloc[i,1] = jang
            cho_word = re.compile('제 *\d+ *조')   
            if re.match(cho_word, df.iloc[i,0]): # 만약 내용에 조가 포함되어 있으면,
                cho = df.iloc[i,0]
            if cho: # 만약 내용에 장이 포함되어 있으면,
                df.iloc[i,2] = cho
        drop_row = []
        for i in range(len(df)):
            if re.match(jang_word, df.iloc[i,0]) or re.match(cho_word, df.iloc[i,0]):
                drop_row.append(i)
        df.drop(drop_row, axis=0, inplace=True)
        df = df[['장', '조', '내용']]

        cho = 'init'
        for i in range(len(df)):
            if cho != df.iloc[i, 1]: # 조가 바뀌었으면
                stack = []
                cho = df.iloc[i, 1] # 조 갱신
                attach = [df.iloc[i, 2]]
                stack.append(chk_char(df, i))
            else: # 조가 안 바뀌었는데
                if chk_char(df, i) != stack[0]: # 글머리 기호가 바뀌었으면
                    if len(stack) > 1:
                        if chk_char(df, i) != stack[1]: # 글머리 기호가 또 바뀌었으면
                            df.iloc[i, 2] = attach[0] + attach[1] + df.iloc[i, 2]
                        else: # 글머리 기호가 이번엔 안 바뀌었으면
                            attach.pop()
                            attach.append(df.iloc[i, 2])
                            df.iloc[i, 2] = attach[0] + df.iloc[i, 2]
                    else:
                        stack.append(chk_char(df, i))
                        attach.append(df.iloc[i, 2])
                        df.iloc[i, 2] = attach[0] + df.iloc[i, 2]
                else:
                    attach = [df.iloc[i, 2]]
                    stack = [stack[0]]
        os.makedirs(path_csv,exist_ok=True)
        df.to_csv(join(path_csv, f'{name}.csv'), index=False, encoding="utf-8-sig")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Input dataset', dest='path_txt', default='./crawl_txt/')
    parser.add_argument('-s', '--save_path', help='Seperated data path', dest='path_csv', default='./crawl_csv/')

    args = parser.parse_args()
    path_txt = args.path_txt
    path_csv = args.path_csv

    seperate(path_txt, path_csv)
    
if __name__ == '__main__':
    main()