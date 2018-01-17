import numpy as np
import gensim as g
import jieba

def Word2vec(model_name):
    # load training data
    data = list()
    dataset_len = list()
    for i in range(5):
        with open('./Data/training_data/'+str(i+1)+'_train.txt', 'r') as f:
            d = list(f)
            data += d
            dataset_len.append(len(d))
    # data pre-processing
    for i in range(len(data)):
        data[i] = data[i].replace('\n', '').replace('.', '').replace('，', '')
    # word segmentation
    jieba.set_dictionary('./libs/dict_new.txt')
    sent_seg = [list(jieba.cut(data[i])) for i in range(len(data))]    
    # word2vec model training (nce loss)
    model = g.models.Word2Vec(sent_seg, window=3, size=4096, max_vocab_size=15000, min_count=1, iter=10)
    # vocas evaluation
    vocas_ = dict()
    for i in range(len(sent_seg)):
        for j in range(len(sent_seg[i])):
            if sent_seg[i][j] in vocas_:
                vocas_[sent_seg[i][j]] += 1
            else:
                vocas_[sent_seg[i][j]] = 1
    in_voca = 0
    not_in_voca = 0
    for k in vocas_.keys():
        if k in model.wv.vocab:
            in_voca += vocas_[k]
        else:
            not_in_voca += vocas_[k]
    print('Original words in gensim voca:', in_voca, ', ratio:', in_voca/(in_voca+not_in_voca),
          '\nOriginal words not in gensim voca:', not_in_voca, ', ratio:', 1-in_voca/(in_voca+not_in_voca))
    # word2vec lookup dictionary
    word2vec = {voca:model.wv.word_vec(voca) for voca in model.wv.vocab.keys()}
    word2vec['<unk>'] = np.zeros_like(word2vec['在'])
    np.save('./model/'+model_name+'/word2vec_gensim', word2vec)    
    return word2vec