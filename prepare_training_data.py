# 準備訓練資料進行文本分類的 data ai:800, info:533, fintech:400
import jieba
import random
import os
from datetime import datetime
from pymongo import MongoClient

def remove_whitespace(str1, str2):
    tmp = str1.replace('\n', '')
    tmp = str1.replace('\r', '')
    tmp = tmp.replace('\t', '')
    str_list = str2.split('(function()')
    tmp2 = str_list[0].replace('\n', '')
    tmp2 = tmp2.replace('\r', '')
    tmp2 = tmp2.replace('\t', '')

    return tmp2

def write_into_txt_file(f, res_list, label):
    '''
    By document
    text format should be as follow
    <label>tab<shorttext>
    '''
    count = 1
    for idx in res_list:
        f.write(label)
        f.write('<check_label>')
        title_content = remove_whitespace(idx['post_title'], idx['post_content'])
        f.write(title_content)
        f.write('\n')
        count += 1


def create_train_txt():
    # 連接資料庫
    client = MongoClient()
    db = client.text_classify
    orange = db.test_msg
    # 產生訓練的txt
    NBA_output_file_name = 'orange_data/NBA_train_data.txt'
    Baseball_output_file_name = 'orange_data/Baseball_train_data.txt'
    PC_Shopping_output_file_name = 'orange_data/PC_Shopping_train_data.txt'
    MobileComm_output_file_name = 'orange_data/MobileComm_train_data.txt'
    cat_output_file_name = 'orange_data/cat_train_data.txt'
    dog_output_file_name = 'orange_data/dog_train_data.txt'
    

    # 設定分類的標籤
    NBA_label = 'NBA'
    MobileComm_label = 'MobileComm'
    PC_Shopping_label = 'PC_Shopping'
    Baseball_label = 'Baseball'
    cat_label = 'cat'
    dog_label = 'dog'

    f1 = open(NBA_output_file_name, 'w')
    f2 = open(Baseball_output_file_name, 'w')
    f3 = open(MobileComm_output_file_name, 'w')
    f4 = open(PC_Shopping_output_file_name, 'w')
    f5 = open(cat_output_file_name, 'w')
    f6 = open(dog_output_file_name, 'w')


    # retrive all data that current exist in message
    NBA_related = list(orange.find({'post_type':'NBA'}).limit(10000))
    Baseball_related = list(orange.find({'post_type':'Baseball'}).limit(10000))
    MobileComm_related = list(orange.find({'post_type':'MobileComm'}).limit(10000))
    PC_Shopping_related = list(orange.find({'post_type':'PC_Shopping'}).limit(10000))
    cat_related = list(orange.find({'post_type':'cat'}).limit(10000))
    dog_related = list(orange.find({'post_type':'dog'}).limit(10000))

    print(len(NBA_related))
    print(len(Baseball_related))
    print(len(dog_related))
    print(len(PC_Shopping_related))
    print(len(MobileComm_related))
    print(len(cat_related))

    random_num = len(NBA_related)
    NBA_random_related_list = random.sample(NBA_related, random_num)
    write_into_txt_file(f1, NBA_random_related_list, NBA_label)

    random_num = len(Baseball_related)
    Baseball_random_related_list = random.sample(Baseball_related, random_num)
    write_into_txt_file(f2, Baseball_random_related_list, Baseball_label)

    random_num = len(MobileComm_related)
    MobileComm_random_related_list = random.sample(MobileComm_related, random_num)
    write_into_txt_file(f3, MobileComm_random_related_list, MobileComm_label)

    random_num = len(PC_Shopping_related)
    PC_Shopping_random_related_list = random.sample(PC_Shopping_related, random_num)
    write_into_txt_file(f4, PC_Shopping_random_related_list, PC_Shopping_label)

    random_num = len(cat_related)
    cat_random_related_list = random.sample(cat_related, random_num)
    write_into_txt_file(f5, cat_random_related_list, cat_label)

    random_num = len(dog_related)
    dog_random_related_list = random.sample(dog_related, random_num)
    write_into_txt_file(f6, dog_random_related_list, dog_label)

    f1.close()
    f2.close()
    f3.close() 
    f4.close() 
    f5.close() 
    f6.close() 
    
    client.close()


if __name__ == '__main__':
    create_train_txt()