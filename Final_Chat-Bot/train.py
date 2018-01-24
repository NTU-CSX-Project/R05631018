from libs.word2vec import Word2vec
from libs.data_loader import loader
from libs.seq2seq import seq2seq_chatbot
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
import jieba
import os
import time

model_name = 'test2'
sent_len = 10

# new model
_ = os.system('mkdir ./model/'+model_name)

# new word2vec model
word2vec_lookup = Word2vec(model_name)
dloader = loader(word2vec_lookup, mode='new', model_name=model_name)

# pre-trained word2vec model
# dloader = loader(mode='pre_trained', model_name=model_name, sent_len=sent_len)

# loading the training data and pre-processing
dloader.data_loading(sent_len=sent_len)

# hyperparameters
np.set_printoptions(precision=2)
jieba.set_dictionary('./libs/dict_new.txt')
vec_size = len(dloader.word2vec_lookup['<unk>'])
oneHot_size = dloader.voca_size
enc_len = sent_len
dec_len = enc_len
n_layer1 = 512
l_r = 1e-3
epoch = 8
batch_size = 128
n_hiddens = 2
rnn_cell = tf.contrib.rnn.BasicLSTMCell
op = tf.train.AdamOptimizer
max_gradient_norm = 1

# seq2seq initializing
s2s = seq2seq_chatbot(oneHot_size=oneHot_size,
                      l_r=l_r,
                      vec_list=dloader.vec_list,
                      vec_size=vec_size,
                      enc_len=enc_len,
                      dec_len=dec_len,
                      n_layer1=n_layer1,
                      n_hiddens=n_hiddens,
                      model_name=model_name)
# encoder
word_vec, encoder_output, encoder_state, go = s2s.encoder(rnn_cell=rnn_cell)
# decoder
word_target, word_outputs, Attention_weights = s2s.decoder_attention(rnn_cell=rnn_cell,
                                                                     encoder_output=encoder_output,
                                                                     encoder_state=encoder_state,
                                                                     go=go)
# loos function
loss_all = s2s.loss_func(word_outputs=word_outputs,
                         word_target=word_target)
# optimizer and update step
update_step = s2s.optimizer(op=op,
                            loss_all=loss_all,
                            max_gradient_norm=max_gradient_norm)

# functions for predicting next sentences
def next_sent(inp):
    inp = list(jieba.cut(inp))
    for j in range(len(inp)):
        if inp[j] in dloader.word2vec_lookup:
            pass
        else:
            inp[j] = '<unk>'
    if len(inp) > sent_len:
        inp = inp[0:sent_len]
    ends = ['<end>' for k in range(sent_len-len(inp))]
    inp += ends
    print(inp)
    batch_x = np.array([[dloader.word2vec_lookup[inp[i]] for i in range(len(inp))]])
    batch_y = np.array([[0 for i in range(dec_len)]])
    resp = list()
    for k in range(dec_len):
        feed_dict = {word_vec[t]: batch_x[:, t] for t in range(enc_len)}
        feed_dict.update({word_target[t]: batch_y[:, t] for t in range(dec_len)})
        
        resp.append(sess.run([word_outputs], feed_dict=feed_dict)[0][k])
        batch_y[0][k] = np.argmax(resp[k])
    print(np.array(sess.run(Attention_weights[0:enc_len], feed_dict=feed_dict)).reshape([-1]))
#     atten_w = np.array(sess.run(Attention_weights, feed_dict=feed_dict)).reshape([-1])
#     atten_avg = list()
#     for i in range(enc_len):
#         atten_avg.append(np.mean([atten_w[i+j*enc_len] for j in range(dec_len)]))
#     print(np.array(atten_avg))
    sent = [dloader.oneHotPos2word[np.argmax(w)] for w in resp]
    return sent
def next_n_sent(inp, n):
    sents = list()
    for i in range(n):
        resp = next_sent(inp)
        sents.append(resp)
        inp = ''.join(sents[i]).replace('<end>', '')
    return sents

# training
init = tf.global_variables_initializer()
ls = list()
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    print('Initial state:')
    for epo in range(epoch):
        t_start = time.time()
        inp = '到現在我還是很懷疑'
        r = next_n_sent(inp, 5)
        for j in range(len(r)):
            print(''.join(r[j]))
        l = list()
        for i in range(int(len(dloader.sent_seg)/batch_size)):
            batch_x = dloader.batch_X(i*batch_size, batch_size)
            batch_y = dloader.batch_Y(i*batch_size, batch_size)
            feed_dict = {word_vec[t]: batch_x[:, t] for t in range(enc_len)}
            feed_dict.update({word_target[t]: batch_y[:, t] for t in range(dec_len)})
            _, lo = sess.run([update_step, loss_all], feed_dict=feed_dict)
            l.append(lo)
            # show the testing every 800 iters
            if (i+1)%800 == 0:
                print('Epoch:', epo+1, 
                      ', iter:', i+1, 
                      ', loss:', np.mean(l))
                print('Input:'+''.join(inp))
                sent = next_sent(inp)
                print('Output:', ''.join(sent))
                print('------------------------------------------------------------------------------')
        ls.append(np.mean(l))
        #-----test-----
        inp = '改天要麻煩你一件事'
        resp = next_n_sent(inp, 5)
        print('Input: '+''.join(inp))
        print('Output: ')
        for j in range(len(resp)):
            print(''.join(resp[j]))
        # plot the loss
        plt.plot(np.arange(len(l)), l)
        plt.title('Loss (epoch'+str(epo+1)+')')
        plt.xlabel('iters')
        plt.ylabel('loss')
        plt.show()
        t_end = time.time()
        print('--------------------------Time span:', t_end-t_start, '--------------------------')
    s2s.save_model(sess=sess,
                   saver=saver,
                   name='test')
    s2s.save_model_params_meta()
    # plot the loss
    plt.plot(np.arange(len(ls))+1, ls)
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
