# 10-fold cross_validation 用學姊的nn模型
# 準備 training data 及 testing data 且比例 9:1
import re
import jieba
import numpy as np
import tensorflow
import keras
from gensim.models import Word2Vec
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
# import xlwt
# import xlrd
import random
import matplotlib.pyplot as plt


# 先載入word2vec model
model_path = 'word2vec.bin'
model = Word2Vec.load(model_path)
jieba.set_dictionary('jieba_dict/dict.txt.big')

word2idx = {"_PAD": 0} # 初始化 [word : token] 字典，後期 tokenize 為embedding層的字典
vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
# 將embedding的參數調好
embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
print(len(vocab_list))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1
    embeddings_matrix[i + 1] = vocab_list[i][1]
print("load finish!")

# 將 word2vec 模型傳給nn
EMBEDDING_DIM = 300 #詞向量維度
embedding_layer = Embedding(len(embeddings_matrix), EMBEDDING_DIM, input_length=250, weights=[embeddings_matrix], trainable=False)

def isset(v): 
    try : 
        type (eval(v)) 
    except : 
        return  0  
    else : 
        return  1  

def remove_symbols(raw_documents):
    regex = re.compile(u"[\u4e00-\u9fff a-z A-Z]+")    # only keep chinese and english
    result = []
    for sen in raw_documents:
        after_regex = ''
        sen = re.findall(regex, sen)
        for word in sen:
            if word != '':
                after_regex += word
        result.append(after_regex)
    return result

def build_model(layer=2, nodes=300, data_dim=300):
    print(data_dim)
    modelA = Sequential()
    modelA.add(embedding_layer)
    modelA.add(LSTM(256, return_sequences=True, input_shape=(data_dim, nodes)))
    modelA.add(Dropout(0.5))
    for i in range(layer-1):
        modelA.add(Dense(512, init="uniform"))
        modelA.add(Activation('relu'))
        modelA.add(Dropout(0.5))
    modelA.add(Flatten())
    modelA.add(Dense(6, kernel_initializer="uniform"))
    modelA.add(Activation('softmax'))

    modelA.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelA

def process_train_test(file_name):
    data_file = open(file_name, 'r')
    data_lines = data_file.readlines()

    train_data = []
    test_data = []
    for line in data_lines:
        train_data.append(line)

    test_data = train_data[int(len(train_data)*1):]
    train_data = train_data[:int(len(train_data)*1)]

    return train_data, test_data

# 分別取ai, info_security及fintech，然後取9:1做training與testing
Baseball_train_data, Baseball_test_data = process_train_test('orange_data/Baseball_train_data.txt')
cat_train_data, cat_test_data = process_train_test('orange_data/cat_train_data.txt')
dog_train_data, dog_test_data = process_train_test('orange_data/dog_train_data.txt')
MobileComm_train_data, MobileComm_test_data = process_train_test('orange_data/MobileComm_train_data.txt')
NBA_train_data, NBA_test_data = process_train_test('orange_data/NBA_train_data.txt')
PC_Shopping_train_data, PC_Shopping_test_data = process_train_test('orange_data/PC_Shopping_train_data.txt')



# 加到一塊，亂數排列
all_train_data_random = Baseball_train_data + cat_train_data + dog_train_data + MobileComm_train_data + NBA_train_data + PC_Shopping_train_data
all_train_data_random = random.sample(all_train_data_random, len(all_train_data_random))

# 整理train data
all_train_data = []
all_label = []

for line in all_train_data_random:
    temp_line = line.split('<check_label>')
    all_label.append(temp_line[0])
    all_train_data.append(temp_line[1])
    
# 正規表示式，只留中文與英文   
after_remove_symbols = remove_symbols(all_train_data)

# 停用詞處理
stopWords = []
remainderWords = []
all_text_after_preprocess = []


with open('jieba_dict/stop_words.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopWords.append(data)

print('stop len:', stopWords[0])
count_word = []
count = 0
for i in range(0, len(after_remove_symbols)):
    temp_str=""
    temp_text = jieba.lcut(after_remove_symbols[i])
    remainderWords = []
    for word in temp_text:
        if word not in stopWords:
            remainderWords.append(word)
            # if word not in count_word:
            #     count_word.append(word)
    # remainderWords = list(filter(lambda a: a not in stopWords and a != '\n', temp_text))
    temp_str = ' '.join(remainderWords)
    all_text_after_preprocess.append(temp_str)

print('total:', len(count_word))    
for i in range(len(all_label)):
    if all_label[i] == "Baseball":
        all_label[i] = 0
    elif all_label[i] == "cat":
        all_label[i] = 1
    elif all_label[i] == 'dog':
        all_label[i] = 2
    elif all_label[i] == 'MobileComm':
        all_label[i] = 3
    elif all_label[i] == 'NBA':
        all_label[i] = 4
    elif all_label[i] == 'PC_Shopping':
        all_label[i] = 5
        

print('label 個數：', len(all_label))
print('資料個數', len(all_text_after_preprocess))
# print(all_text_after_preprocess[0:10], all_label[0:10])

train_data = all_text_after_preprocess
train_data_label = to_categorical(all_label, 6)
print(train_data_label[0])

tk = Tokenizer()
tk.fit_on_texts(train_data)
index_list = tk.texts_to_sequences(train_data)
x_train = pad_sequences(index_list, maxlen=250)
print(x_train.shape[0], x_train.shape[1])


model = build_model(layer=2, nodes=300, data_dim=x_train.shape[1])
model.summary()

history = model.fit(x_train, train_data_label, batch_size=500, epochs=100, verbose=2, validation_split=0.1)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("accuracy.png")
plt.show()

model.save('lstm_model.h5')
