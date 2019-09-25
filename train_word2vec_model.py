# 訓練 word2vec model
import re
import jieba
import codecs
import logging
from gensim.models import word2vec

# 載入停用辭典
def load_stopword():
    stop = []
    fle = open('jieba_dict/stop_words.txt', 'r')
    stop = fle.read()
    stop = stop.splitlines()
    return stop

# 利用jieba斷詞，產生斷詞文檔'data_seg.txt'
def sen_jieba(after_remove):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # jieba custom setting.
    jieba.set_dictionary('jieba_dict/dict.txt.big')
    
    # load stopwords set
    stopword_set = load_stopword()
    # with open('jieba_dict/stop_words.txt','r') as stopwords:
    #     for stopword in stopwords:
    #         stopword_set.add(stopword.strip('\n'))
    
    print('stop len:', len(stopword_set))
    output = open('data_seg.txt', 'w', encoding='utf-8')
    for texts_num, line in enumerate(after_remove):
        words = jieba.lcut(line)
        temp_word = []
        for word in words:
            if word not in stopword_set and word != ' ':
                temp_word.append(word)
        temp_sentence = ' '.join(temp_word)
        output.write(temp_sentence)
        output.write('\n')

        if (texts_num + 1) % 10000 == 0:
            logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
    output.close()

# 利用正規表示式只保留文章中文與英文的部份
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

#read dpcument
file = open('word2vec_corpus.txt', 'r')
raw_documents = []
for line in open('word2vec_corpus.txt', 'r'):
    line = file.readline().strip()
    if len(line) != 0 :
        raw_documents.append(line)

#remove symbols, only keep chinese and english
after_remove = remove_symbols(raw_documents)
#use jieba to seg word
sen_jieba(after_remove)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence("data_seg.txt")
# 訓練 word2vec_model
model = word2vec.Word2Vec(sentences, size=300, min_count=1)

model.save("word2vec.bin")
