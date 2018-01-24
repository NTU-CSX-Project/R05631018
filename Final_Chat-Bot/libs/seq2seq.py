import tensorflow as tf
import numpy as np
import time

class seq2seq_chatbot(object):
    def __init__(self, oneHot_size=None, l_r=None, vec_list=None, vec_size=None, enc_len=None,
                 dec_len=None, n_layer1=None, n_hiddens=None, model_name=None, load_model=False):
        if load_model:
            pass
        else:
            self.oneHot_size = oneHot_size
            self.l_r = l_r
            self.vec_list = vec_list
            self.vec_size = vec_size
            self.enc_len = enc_len
            self.dec_len = dec_len
            self.n_layer1 = n_layer1
            self.params = dict()
            self.params['var_name'] = list()
            self.n_hiddens = n_hiddens
        self.model_name = model_name
    def _embed_lookup_const(self):
        self.vecs = tf.constant(self.vec_list)
    def _weight(self, shape):
        initial = tf.truncated_normal(shape, stddev=1.)
        return tf.Variable(initial)
    def _bias(self, shape):
        initial = tf.constant(1., shape=shape)
        return tf.Variable(initial)
    def encoder(self, rnn_cell):
        # word vectors constant
        self._embed_lookup_const()
        
        # input - word vectors
        word_vec = [tf.placeholder(tf.float32, shape=[None, self.vec_size], name='word_vec') for i in range(self.enc_len)]    

        # encoder input
        w_enc = self._weight([self.vec_size, self.n_layer1])
        b_enc = self._bias([self.n_layer1])
        enc_inp = [tf.nn.dropout(tf.sigmoid(tf.matmul(word_vec[i], w_enc)+b_enc), keep_prob=.5) \
                   for i in range(self.enc_len)]
        enc_inp = [tf.nn.batch_normalization(x=enc_inp[i],
                                             mean=0,
                                             variance=.8,
                                             offset=0,
                                             scale=1,
                                             variance_epsilon=.001,
                                             name='enc_inp')\
                   for i in range(self.dec_len)]

        # encoder
        with tf.variable_scope('enc'):
            cells = [rnn_cell(self.n_layer1) for i in range(self.n_hiddens)]
            encoder_cell = tf.contrib.rnn.MultiRNNCell(cells)
            encoder_output, encoder_state = tf.contrib.rnn.static_rnn(encoder_cell,
                                                                      enc_inp,
                                                                      dtype=tf.float32)
            w_enc_out = self._weight([self.n_layer1, self.vec_size])
            b_enc_out = self._bias([self.vec_size])
            go = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(encoder_output[-1], w_enc_out), b_enc_out)),
                               keep_prob=.5,
                               name='go')
        
        self.params['var_name'] += ['word_vec']
        self.params['word_vec'] = [word_vec[i].name for i in range(len(word_vec))]
        return word_vec, encoder_output, encoder_state, go
    def decoder_attention(self, rnn_cell, encoder_output, encoder_state, go):
        # training target - word one-hot
        word_target = [tf.placeholder(tf.int32, shape=[None], name='word_target') for i in range(self.dec_len)]

        # one-hot to word vectors
        target = [go] + [tf.nn.embedding_lookup(self.vecs, word_target[i]) for i in range(self.dec_len-1)]

        # decoder input
        w_dec = self._weight([self.vec_size, self.n_layer1])
        b_dec = self._bias([self.n_layer1])
        dec_inp = [tf.nn.dropout(tf.nn.relu(tf.matmul(target[i], w_dec)+b_dec), keep_prob=.5, name='dec_inp') \
                   for i in range(self.dec_len)]
        dec_inp = [tf.nn.batch_normalization(x=dec_inp[i],
                                             mean=0,
                                             variance=.8,
                                             offset=0,
                                             scale=1,
                                             variance_epsilon=.001,
                                             name='dec_inp')\
                   for i in range(self.dec_len)]

        # decoder and attention
        with tf.variable_scope('dec'):
            cells = [tf.contrib.rnn.BasicLSTMCell(self.n_layer1) for i in range(self.n_hiddens)]
            decoder_cell = tf.contrib.rnn.MultiRNNCell(cells)
            decoder_output, _ = tf.contrib.rnn.static_rnn(decoder_cell,
                                                          dec_inp,
                                                          initial_state=encoder_state)
            # attention
            Luong_scores = list()
            Attention_weights = list()
            Context_vectors = list()
            Attention_vectors = list()
            w_scores = self._weight([self.n_layer1*2, 1])
            b_scores = self._bias([1])
            w_atten_vecs = self._weight([self.n_layer1*2, self.oneHot_size])
            b_atten_vecs = self._bias([self.oneHot_size])
            for i in range(self.dec_len):
                dec_o = decoder_output[i]
                # score func [enc_len, batch_size, 1]
                scores = [tf.nn.relu(tf.add(tf.matmul(tf.concat([encoder_output[j], dec_o], axis=1),
                                                      w_scores),
                                            b_scores), name='score')
                          for j in range(self.enc_len)]
                # attention weight [enc_len, batch_size, 1]
                sum_scores = tf.add_n(scores, name='sum_scores')+0.1 # avoid nan (dividing by 0)
                weights = [tf.divide(scores[j]+(0.1/self.enc_len), sum_scores, name='atten_weight')
                           for j in range(self.enc_len)]
                # context vector
                context_vec = tf.add_n([tf.multiply(weights[j], encoder_output[j]) for j in range(self.enc_len)],
                                       name='context_vec')
                # attention vector
                atten_vec = tf.add(tf.matmul(tf.concat([context_vec, dec_o], axis=1), w_atten_vecs),
                                   b_atten_vecs,
                                   name='word_output')
                Luong_scores.append(scores)
                Attention_weights.append(weights)
                Context_vectors.append(context_vec)
                Attention_vectors.append(atten_vec)
        word_outputs = Attention_vectors
        self.params['var_name'] += ['word_target']
        self.params['word_target'] = [word_target[i].name for i in range(len(word_target))]
        self.params['var_name'] += ['word_outputs']
        self.params['word_outputs'] = [word_outputs[i].name for i in range(len(word_outputs))]
        self.params['var_name'] += ['Attention_weights']
        self.params['Attention_weights'] = [Attention_weights[i][j].name for i in range(len(Attention_weights))
                                            for j in range(len(Attention_weights[i]))]
        return word_target, word_outputs, self.params['Attention_weights']
    def loss_func(self, word_outputs, word_target):
        loss_all = 0
        for y_hat, y_real in zip(word_outputs, word_target):
            loss_all += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=y_real)
        loss_all = tf.reduce_mean(loss_all/self.dec_len, name='loss_all')
        self.params['var_name'] += ['loss_all']
        self.params['loss_all'] = [loss_all.name]
        return loss_all
    def optimizer(self, op, loss_all, max_gradient_norm):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss_all, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
        optimizer = op(self.l_r)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        self.params['var_name'] += ['update_step']
        self.params['update_step'] = [update_step.name]
        return update_step
    def save_model_params_meta(self):
        return np.save('./model/'+self.model_name+'/model_params_meta', self.params)
    def save_model(self, sess, saver, name=time.strftime('%Y.%m.%d_%H%M%S')):
        return saver.save(sess, './model/'+self.model_name+'/'+name+'_chatbot')
    def load_model_params_meta(self):
        self.params = np.load('./model/'+self.model_name+'/model_params_meta.npy')[None][0]
        return self.params['word_vec'], self.params['word_target'], self.params['word_outputs'], self.params['Attention_weights'], self.params['loss_all'], self.params['update_step']
    def load_model(self, sess, name=''):
        print('Restoring model...')
        path = './model/'+self.model_name+'/'+name+'_chatbot.meta'
        saver = tf.train.import_meta_graph(path)
        saver.restore(sess, path[:-5])
        return print('done')