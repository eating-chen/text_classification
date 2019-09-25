# 訓練數據準備
import jieba
import random
import os
from datetime import datetime
from pymongo import MongoClient

def remove_whitespace(str1, str2):
    tmp = str1.replace('\n', '')
    tmp = str1.replace('\r', '')
    tmp = tmp.replace('\t', '')
    tmp2 = str2.replace('\r', '')
    tmp2 = tmp2.replace('\t', '')
    return tmp + tmp2

def write_into_txt_file(f, res_list):
    '''
    By document
    text format should be as follow
    <label>tab<shorttext>
    '''
    count = 1
    for idx in res_list:
        title_content = remove_whitespace(idx['post_title'], idx['post_content'])
        f.write(title_content)
        f.write('\n')
        count += 1

def create_train_txt():
    # set the db connection
    client = MongoClient()
    db = client.text_classify
    _message = db.test_msg
    # initiate data
    # file setting
    output_file_name = 'word2vec_corpus.txt'
    f = open(output_file_name, 'w')

    # retrive all data that current exist in message
    _res = list(_message.find({}))

    write_into_txt_file(f, _res)

    f.close()
    client.close()

if __name__ == '__main__':
    create_train_txt()
