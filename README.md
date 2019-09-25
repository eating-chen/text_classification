# text_classification

work 說明
---
- 爬取ptt的版：Baseball, cat, dog, MobileComm, NBA, PC_Shopping 
- 各版各有10000篇文章
- 存至 mongodb
- 用word2vec轉成詞向量後，用lstm進行文本分類

套件
---
- keras

程式說明
---
- prepare_word2vec_corpus.py 用來產生訓練 word2vec 的文檔
- train_word2vec_model.py 用來訓練 word2vec model
- prepare_training_data.py 準備六個版的data
- train_nn.py 訓練lstm, 以 9:1的方式訓練

結果
---
![](https://github.com/eating-chen/text_classification/blob/master/accuracy.png)
