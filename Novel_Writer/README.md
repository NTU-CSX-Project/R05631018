## Novel Writer

 - Materials : 神雕俠侶 (romance_condor_heroes.txt)

 - Model : Seq2seq model with encoder:decoder = 50:50 <br>
![png](./imgs/seq2seq.png)

## Procedure

 - Chinese segmentation: by jieba with big-5 dictionary (dict.txt.big)

 - Word2vec: by gensim, length = 64

 - One-hot encoding: each word is a class

 - Model: seq2seq model with LSTM

 - Epoch: 3

 - Optimizer: RMSprop

 -- Result:
