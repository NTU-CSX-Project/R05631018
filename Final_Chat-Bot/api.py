import json
from bottle import run, post, request, response, get, route
import time
from libs.word2vec import Word2vec
from libs.data_loader import loader
from libs.seq2seq import seq2seq_chatbot
import tensorflow as tf
import numpy as np
import jieba
import os

np.set_printoptions(precision=2)
model_name = 'test2'
dloader = loader(mode='pre_trained', model_name=model_name)

sess = tf.Session()
s2s = seq2seq_chatbot(load_model=True, model_name=model_name)
s2s.load_model(sess=sess, name='test')
word_vec, word_target, word_outputs, Attention_weights, loss_all, update_step = s2s.load_model_params_meta()

# functions for predicting next sentences
sent_len = len(word_vec)
enc_len = len(word_vec)
dec_len = len(word_target)
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
    atten_w = np.array(sess.run(Attention_weights, feed_dict=feed_dict)).reshape([-1])
    atten_avg = list()
    for i in range(enc_len):
        atten_avg.append(np.mean([atten_w[i+j*enc_len] for j in range(dec_len)]))
    print(np.array(atten_avg))
    sent = [dloader.oneHotPos2word[np.argmax(w)] for w in resp]
    return sent
def next_n_sent(inp, n):
    sents = list()
    for i in range(n):
        resp = next_sent(inp)
        sents.append(resp)
        inp = ''.join(sents[i]).replace('<end>', '')
    return sents

@route('/<path>', method = 'POST')
def process(path):
    if path == 'sent':
        enc = json.JSONEncoder()
        dec = json.JSONDecoder()
        print('from', request.remote_addr, path)
        print('input json:', type(request.json), request.json, type(request.json) is not dict)
        print('content type:', request.content_length, request.content_type)
        
        if type(request.json) is not str:
            req_json = str(request.json)
        else:
            req_json = request.json
        req_json = req_json.replace('\'', '"')
        sent = dec.decode(req_json)
        sent = sent['sent']
        print(sent)
        #_resp = next_n_sent(sent, 2)
        _resp = next_sent(sent)
        resp = ''.join(_resp).replace('<end>', '')+' .'
        #resp += ''.join(_resp[1]).replace('<end>', '')
        return_json = str({"resp":resp}).encode('utf-8')
        print(resp, type(resp), return_json, type(return_json))
        return return_json
    else:
        return None

run(host='140.112.94.35', port=8186, debug=False)