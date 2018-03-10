#coding=utf-8
#-*- coding: UTF-8 -*-
import sys

"""python3"""

import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Lambda, Reshape, Activation, Input, Embedding, GRU, LSTM
from keras.layers.merge import concatenate, dot
from keras.models import Model
import numpy as np

import os
import codecs
import random

global param
param = {}

param['train_batch_size'] = 64
param['test_batch_size'] = 128
param['steps_per_test'] = 1000
param['most_epoches'] = 50

param['sample_per_test'] = param['steps_per_test'] * param['train_batch_size']

if __name__ == '__main__':
    param['model_path'] = 'model/'
    param['log_path'] = None

    if not os.path.exists(param['model_path']) :
        os.mkdir(param['model_path'])
    if param['log_path'] is None :
        param['log_path'] = os.path.join(param['model_path'], 'log')
    param['log_file'] = codecs.open(param['log_path'], 'w', 'utf-8')
        
"""
rnnlm model
"""
class C_rnnlm() :
    def __init__(self, word_vector_matrix) :
        self.word_vector_matrix = word_vector_matrix
        print('word_vector_matrix_shape', np.shape(word_vector_matrix))

        self.decoder_words_num = 40000
        self.encoder_words_num = 40000
        self.hidden_units      = 300
        self.embedding_length  = 100

        self.optimizer='rmsprop'
        self.dis_loss = lambda y_true, y_pred: K.mean((y_pred-y_true), axis = -1)
        """
        这个损失函数就是预测值减去真实值
        而事实上，一般真实值输进去的都是零，所以loss就直接对预测值计算均值
        """
    def get_model(self) :
        """  定义损失函数的计算方法  """
        def count_loss(none_place, input1, input2) :
            dot = K.batch_dot(input1, input2, axes=(2,2))
            # (?, 100, 100)
            log_p = K.log(dot)
            # (?, 100, 100)
            trace = tf.trace(log_p) 
            # (?,)
            results = -K.expand_dims(trace, axis=-1)
            # (?, 1)
            return results

        """  定义网络层  """
        def my_embeddings_initializer(shape, dtype=None):
            return self.word_vector_matrix
        self.embedding = Embedding(input_dim=self.encoder_words_num, output_dim=self.embedding_length,
                                   embeddings_initializer=my_embeddings_initializer, input_length=(None,), mask_zero=True)
        self.decoder_rnn = LSTM(units=self.hidden_units, return_sequences=True, return_state=True)
        self.softmax_dense = Dense(self.decoder_words_num, activation='softmax')
        self.ta_onehot = Lambda(lambda x : K.one_hot(x, self.decoder_words_num), 
                            output_shape=lambda input_shape: tuple(list(input_shape)+[self.decoder_words_num]))

        """  定义网络结构  """
        input_seq  = Input(shape=(None,), dtype='int32', name='input')
        output_seq = Input(shape=(None,), dtype='int32', name='output')
        # input_seq  = Input(shape=(100,), dtype='int32', name='input')
        # output_seq = Input(shape=(100,), dtype='int32', name='output')
        # embedding: 300
        # (?, 100) (?, 100)

        input_emb = self.embedding(input_seq)
        # (?, 100, 300)
        decoder_outputs, _1, _2 = self.decoder_rnn(input_emb)
        # (?, ?, 300) (?, 300) (?, 300)
        decoder_results = self.softmax_dense(decoder_outputs)
        # (?, 100, 30000)

        target_out = self.ta_onehot(output_seq)
        # (?, 100, 30000)

        """  语言模型训练  """
        loss_results = Lambda(lambda x : count_loss(x, input1=decoder_results, input2=target_out), lambda x:(None, 1))(decoder_results)
        # (?, 1)

        self.language_model = Model([input_seq, output_seq], loss_results)
        self.language_model.compile(optimizer='rmsprop', loss=self.dis_loss)

        """  语言模型生成  """
        decoder_state_input_h = Input(shape=(self.hidden_units,), dtype='float32', name='decoder_state_input_h')
        decoder_state_input_c = Input(shape=(self.hidden_units,), dtype='float32', name='decoder_state_input_c')

        decoder_outputs_gen, state_h, state_c = self.decoder_rnn(input_emb, initial_state=[decoder_state_input_h, decoder_state_input_c])
        # (?, ?, 300) (?, 300) (?, 300)
        decoder_outputs_gen_single, state_h_s, state_c_s = self.decoder_rnn(input_emb)
        # (?, ?, 300)
        decoder_results_gen = self.softmax_dense(decoder_outputs)
        # (?, 100, 30000)
        decoder_results_gen_single = self.softmax_dense(decoder_outputs_gen_single)
        # (?, 100, 30000)

        self.language_model_gen = Model([input_seq, decoder_state_input_h, decoder_state_input_c], [decoder_results_gen, state_h, state_c])
        self.language_model_gen_single = Model([input_seq], [decoder_results_gen_single, state_h_s, state_c_s])

    def get_possiblity(self, input_seq, init_states=None) :
        if not init_states is None :
            assert len(init_states) == 2
            assert len(input_seq) == len(init_states[0]) == len(init_states[1])
        assert len(input_seq) == 1

        if init_states is None :
            decoder_result, state_h, state_c = self.language_model_gen_single.predict([input_seq])
        else :
            decoder_result, state_h, state_c = self.language_model_gen.predict([input_seq, init_states[0], init_states[1]])

        return decoder_result, [state_h, state_c]

def load_matrix(path, dtype=float) :
    data = codecs.open(path, 'r', 'utf-8').readlines()
    data = [list(map(dtype, t.strip().split(' '))) for t in data]
    data = np.array(data)

    return data

"""
data_proxy
"""
class C_data_proxy :
    def __init__(self) :
        train_in  = load_matrix('tar-in-out/trn.tgt.in', dtype=int)
        train_out = load_matrix('tar-in-out/trn.tgt.out', dtype=int)
        valid_in  = load_matrix('tar-in-out/dev.tgt.in', dtype=int)
        valid_out = load_matrix('tar-in-out/dev.tgt.out', dtype=int)
        print('train_in shape: ', np.shape(train_in))
        print('train_out shape: ', np.shape(train_out))
        print('valid_in shape: ', np.shape(valid_in))
        print('valid_out shape: ', np.shape(valid_out))
        
        assert len(train_in) == len(train_out)
        assert len(valid_in) == len(valid_out)
        self.train_data = [{'in':train_in[i], 'out':train_out[i], 'zero':[0]}for i in range(len(train_in))]
        self.valid_data = [{'in':valid_in[i], 'out':valid_out[i], 'zero':[0]}for i in range(len(valid_in))]
        
        self.train_keys = {'x':['in', 'out'], 'y':['zero']}
    
    def get_train_data(self, data_raw, size=None, seed=42) :
        data = []
        if not size is None :
            random.seed(seed)
            for i in range(size) :
                data.append(random.choice(data_raw))
        else :
            data += data_raw

        data_x, data_y = [], []
        for key in self.train_keys['x'] :
            temp_matrix = np.array([dt[key] for dt in data])
            data_x.append(temp_matrix)
        for key in self.train_keys['y'] :
            temp_matrix = np.array([dt[key] for dt in data])
            data_y.append(temp_matrix)
        return data_x, data_y

"""
train
"""
def train(rnnlm, data_proxy) :
    
    valid_in, valid_out     = data_proxy.get_train_data(data_proxy.valid_data)
    train_s_in, train_s_out = data_proxy.get_train_data(data_proxy.train_data, size=int(param['sample_per_test']/100))

    history = rnnlm.language_model.fit(train_s_in, train_s_out,
                batch_size=param['train_batch_size'], epochs=1, shuffle=True)
    train_loss = history.history['loss'][0]
    test_loss = 0
    

    for i in range(param['most_epoches']) :
        print('epoches : %d'%(i))
        model_path_t = os.path.join(param['model_path'], '%d'%(i))
        rnnlm.language_model.save_weights(model_path_t)

        test_loss = rnnlm.language_model.evaluate(valid_in, valid_out, batch_size=param['test_batch_size'], verbose=1)

        if not param['log_path'] is None :
            param['log_file'].write('epoches : %d\ttrn_loss : %.6f\ttst_loss : %.6f\n'%(i, train_loss, test_loss))
            param['log_file'].flush()
        
        train_in, train_out = data_proxy.get_train_data(data_proxy.train_data, size=param['sample_per_test'], seed=i)
        history = rnnlm.language_model.fit(train_in, train_out,
            batch_size=param['train_batch_size'], epochs=1, shuffle=True)
        train_loss = history.history['loss'][0]
        
    return rnnlm, data_proxy

if __name__ == '__main__':
    # word_vector_matrix = np.random.random([40000, 300])
    word_vector_matrix = load_matrix('vector.matrix.100', dtype=float)

    rnnlm = C_rnnlm(word_vector_matrix=word_vector_matrix)
    rnnlm.get_model()

    # data_proxy = C_data_proxy()

    # train(rnnlm, data_proxy)

    rnnlm.language_model.load_weights('model/6')

    """"""

    p0, state1 = rnnlm.get_possiblity(np.array([[2]]))
    p1, state2 = rnnlm.get_possiblity(np.array([[9]]), state1)
    p2, _ = rnnlm.get_possiblity(np.array([[9]]), state2)
    '''
    P(xi=vi|xi-1, xi-2, ... , x0)

    p1 P(xi|2 9)
    p2 P(xi|2 9 9)
    '''

    print(p0)
    print(p1)
    print(len(p0[0][0]),len(p1[0][0]))
    # print(state1,state2)
    # print(np.shape(p0), np.shape(p1))
    # print(len(state), np.shape(state[0]), np.shape(state[1]))


"""
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS-no-openmp/lib
export CUDA_VISIBLE_DEVICES=-1
"""

    
def decode_cpu(self, data, max_length, early_stop=True, beam_size=5, top_n=None, son_timer=None) :
    """
    这是一个使用CPU进行进行解码的函数，要求batch_size=1，使用beam search

    @data : 所有的输入数据
      @input_sequence  : 输入词序列。这只是用来处理UNK的时候指向使用的。
      @encoder_outputs : 输入词序列的编码序列
      @encoder_states  : 输入词序列的编码状态，实际上形式为[state_h, state_c]
    @max_length      : 最大编码长度，到那就停
    @early_stop      : 是否进行提前停止，即Beam里面所有分支都连续两次解码出终止字符。
    @beam_size       : 维护的最大Beam的大小
    @top_n           : 最终返回概率最大的top_n句话。如果top_n is None（推荐），则返回回收器中所有的话。
    @son_timer       : 一个计时器，如果为None则内部生成一个，否则，可以用于全局时间统计。
    """
    input_sequence  = [data['source_inputs']]
    encoder_outputs = np.array([data['encode_data_outputs']])
    encoder_states  = map(np.array, [[data['encode_data_states_h']], [data['encode_data_states_c']]])

    """
    强行要求输入的batch_size为1
    """
    decoder_ids = range(self.decoder_words_num)
    if son_timer is None :
        son_timer = MyTimer()
    batch_size = len(encoder_outputs)
    assert batch_size == 1
    assert batch_size == len(input_sequence)
    input_sequence = input_sequence[0]


    # 由于回收机制的引入，这一要求未必成立，所以注释: 
    # assert top_n <= beam_size  （前面是不成立的要求）

    # 在实际处理时top_n为None，也并不是返回所有，而是最多9999个。
    if top_n is None :
        top_n = 9999
    son_timer.tick(to_count=True, key='Prepare: ')

    """
    初始化Beam_Search的一些信息

    Beam中存的5条数据：
        0                   初始概率
        input_seq           下一步即将输入的序列，类型为np.array([[int]])
        encoder_states      当前的编码状态，下一步输入，形如：[state_h, state_c]
        []                  decode_history，为之前已经确定的需要的序列，理论上包含input_seq，实际上，考虑到第一时刻的输入为<SOS>，所以初始化为空
        []                  unk_info，为之前解码出的unk，attention指向的是输入序列的第几个。这只针对指向的输入也是UNK时的处理。
    """
    input_seq = [[self.id_sos]]
    input_seq = np.array(input_seq)
    beam_queue = [[0, input_seq, encoder_states, [], []]]
    output_sequence = []
    
    son_timer.tick(to_count=True, key='Prepare_decode_all: ')
    for times in range(max_length) :
        son_timer.tick(to_count=True, key='Decode_start: ')
        # print times, len(beam_queue)
        """新的beam队列，在这一步的解码中进行填充，并最终替换旧的beam队列"""
        beam_queue_new = []

        son_timer.tick(to_count=True, key='Prepare_decode_: ')
        for score, input_seq_t, encoder_states_t, history, unk_info in beam_queue :
            """解码，状态预测，及打包"""
            son_timer.tick(to_count=True, key='Beam_start: ')
            next_state, decoder_state_h, decoder_state_c, attention_results = self.unit_model.predict([input_seq_t, encoder_outputs] + encoder_states_t)
            son_timer.tick(to_count=True, key='Beam_predict: ')
            encoder_states_new = [decoder_state_h, decoder_state_c]
            son_timer.tick(to_count=True, key='Prepare_search: ')

            """对下一时刻状态的初步筛选"""
            next_state = next_state[0][0]
            son_timer.tick(to_count=True, key='Finish_search: list:')
            next_state = zip(next_state, decoder_ids)
            son_timer.tick(to_count=True, key='Finish_search: zip:')
            next_state = LazySorted(next_state, reverse=True)[:beam_size]
            son_timer.tick(to_count=True, key='Finish_search: sorted:')

            for score_t, word_id_t in next_state :
                new_unk_info = [t for t in unk_info]
                """对UNK进行处理"""
                if word_id_t == self.id_unk :
                    attention_results = attention_results[0][0]
                    max_id_zip = np.argmax(attention_results)
                    word_id_t = input_sequence[max_id_zip]

                    """如果指向的input也是unk，就要把attention目标位置存下来，以便后处理时进行还原"""
                    if word_id_t == self.id_unk :
                        new_unk_info.append(max_id_zip)

                    son_timer.tick(to_count=True, key='Finish_search: unk_word:')

                """更新解码信息，并加入新的beam中"""
                new_history = history+[word_id_t]
                assert new_history.count(self.id_unk) == len(new_unk_info)
                beam_queue_new.append([score+math.log(score_t), np.array([[word_id_t]]), encoder_states_new, new_history, new_unk_info])
                son_timer.tick(to_count=True, key='Finish_search: append:')

        """维护新beam大小，现在就只有beam_size个了，并且排好序"""
        # beam_queue_new = LazySorted(beam_queue_new, reverse=True)[:beam_size]
        beam_queue_new = sorted(beam_queue_new, key=lambda x: x[0], reverse=True)[:beam_size]

        """
        回收过程：
        回收标准为beam_queue_new中，最后collection_check_num位（默认为2）均为终止字符。
        满足回收标准的beam中的内容会被丢入回收器中，否则则留给下一时刻的beam
        """
        beam_queue = []
        collection_check_num = 2
        for temp in beam_queue_new :
            temp_history = temp[-2]
            if len(temp_history) >= collection_check_num :
                if all([t in self.stop_marks for t in temp_history[-collection_check_num:]]) :
                    output_sequence.append([temp[-2], temp[-1], temp[0]])
                    continue
            beam_queue.append(temp)

        """
        解码过程终止条件，除了解码到预定步数之外，还有3种情况需要终止：
        1、回收器已经回收足够到top_n的规模
        2、beam中已经为空，即都已经终止回收
        3、设定了early_stop，且所有beam中元素的当前状态都是终止状态。
        """
        if len(output_sequence) >= top_n :
            break
        if len(beam_queue) == 0 :
            break
        if early_stop and all([t[1][0][0] in self.stop_marks for t in beam_queue]) :
            break
    son_timer.tick(to_count=True, key='Finish_Beam: ')

    """
    回收器，也是即将返回的数据结构，组织形式如下：
    [t[-2], t[-1], t[0]] : history, unk_info, score
    即，其中每个元素分别为：历史解码信息，UNK相关的指代，概率分数
    """
    output_sequence += [[t[-2], t[-1], t[0]] for t in beam_queue]
    output_sequence = output_sequence[:top_n]

    son_timer.tick(to_count=True, key='Get_output: ')
    return output_sequence






