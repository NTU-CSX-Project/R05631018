import numpy as np
import jieba

class loader(object):
    def __init__(self, word2vec_lookup=None, mode='pre_trained', model_name=''):
        print('Data loader initializing')
        print('mode: ' + mode)
        self.model_name = model_name
        if mode is 'pre_trained':
            print('Pre-trained dicts loading...')
            self.word2vec_lookup = np.load('./model/'+model_name+'/word2vec_gensim.npy')
            self.word2vec_lookup = self.word2vec_lookup[None][0]
            self.word2vec_lookup['<unk>'] = np.zeros_like(self.word2vec_lookup['在'])
            self.word2vec_lookup['<end>'] = np.ones_like(self.word2vec_lookup['在'])*-1
            self.word2oneHot = np.load('./model/'+model_name+'/word2oneHot.npy')
            self.word2oneHot = self.word2oneHot[None][0]
            self.word2oneHotPos = np.load('./model/'+model_name+'/word2oneHotPos.npy')
            self.word2oneHotPos = self.word2oneHotPos[None][0]
            self.oneHotPos2word = np.load('./model/'+model_name+'/oneHotPos2word.npy')
            self.oneHotPos2word = self.oneHotPos2word[None][0]
            self.oneHotPos2vec_lookup = np.load('./model/'+model_name+'/oneHotPos2vec_lookup.npy')
            self.oneHotPos2vec_lookup = self.oneHotPos2vec_lookup[None][0]
            self.vec_list = np.load('./model/'+model_name+'/vec_list.npy')
            self.voca_size = len(self.word2vec_lookup.keys())
            print('done')
        elif mode is 'new':
            print('Creating new dicts...')
            self.word2vec_lookup = word2vec_lookup
            self.word2vec_lookup['<unk>'] = np.zeros_like(self.word2vec_lookup['在'])
            self.word2vec_lookup['<end>'] = np.ones_like(self.word2vec_lookup['在'])*-1
            self.voca_size = len(self.word2vec_lookup.keys())
            self.word2oneHot = dict()
            self.word2oneHotPos = dict()
            self.oneHotPos2word = dict()
            self.oneHotPos2vec_lookup = dict()
            words = np.array(list(self.word2vec_lookup.keys()))
            # one-hot encoding and position
            for i in range(self.voca_size):
                self.word2oneHot[words[i]] = np.zeros(self.voca_size).astype('int')
                self.word2oneHot[words[i]][i] = 1
                self.word2oneHotPos[words[i]] = i
            # one-hot 2 word and vec
            for key, val in self.word2oneHotPos.items():
                self.oneHotPos2vec_lookup[val] = word2vec_lookup[key]
                self.oneHotPos2word[val] = key
            self.vec_list = [list(self.oneHotPos2vec_lookup[i]) for i in range(len(self.oneHotPos2vec_lookup.keys()))]
            print('Saving new dicts at ./model/'+model_name+'/')
            np.save('./model/'+model_name+'/word2oneHot', self.word2oneHot)
            np.save('./model/'+model_name+'/word2oneHotPos', self.word2oneHotPos)
            np.save('./model/'+model_name+'/oneHotPos2word', self.oneHotPos2word)
            np.save('./model/'+model_name+'/oneHotPos2vec_lookup', self.oneHotPos2vec_lookup)
            np.save('./model/'+model_name+'/vec_list', self.vec_list)
            print('done')
        else:
            print('Initialzing error, mode name?')
    def data_loading(self, sent_len):
        self.sent_len = sent_len
        data = list()
        dataset_len = list()
        for i in range(5):
            with open('./Data/training_data/'+str(i+1)+'_train.txt', 'r') as f:
                d = list(f)
                data += d
                dataset_len.append(len(d))
        # pre-processing
        for i in range(len(data)):
            data[i] = data[i].replace('\n', '').replace('.', '').replace('，', '')
        print('Pre-processing 1')
        jieba.set_dictionary('./libs/dict_new.txt')
        sent_seg = [list(jieba.cut(data[i])) for i in range(len(data))]
        print('Pre-processing 2')
        for i in range(len(sent_seg)):
            # remove unk items
            rm_idx = list()
            for j in range(len(sent_seg[i])):
                if sent_seg[i][j] in self.word2vec_lookup:
                    pass
                else:
                    rm_idx.append(j)
            sent_seg[i] = list(np.delete(sent_seg[i], rm_idx))
            # fix the length        
            if len(sent_seg[i])>(self.sent_len-1):
                sent_seg[i] = sent_seg[i][0:self.sent_len]
            # add end to the end
            ends = ['<end>' for k in range(self.sent_len-len(sent_seg[i]))]
            sent_seg[i] += ends
        print('Pre-processing done.')
        self.sent_seg = sent_seg
        print('sent_seg var is created')
    def batch_X(self, i, batch):
        batch_x = list()
        for s in self.sent_seg[i:i+batch]:
            v = list()
            for w in s:
                v.append(self.word2vec_lookup[w])
            batch_x.append(v)
        batch_x = np.array(batch_x)
        return batch_x
    def batch_Y(self, i, batch):
        batch_y = list()
        for s in self.sent_seg[i+1:i+batch+1]:
            v = list()
            for w in s:
                v.append(np.argmax(self.word2oneHot[w]))
            batch_y.append(v)
        batch_y = np.array(batch_y)
        return batch_y
        