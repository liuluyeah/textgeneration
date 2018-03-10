#coding=utf-8
#-*- coding: UTF-8 -*-
import sys

"""python3"""

import os, codecs, json, math
import random
from tqdm import tqdm
from lazysorted import LazySorted
import numpy as np
from functools import reduce 

def n_gram_seperate(input_path) :
    """
     @param input_path:  输入为词频统计文件，例如trn.tgt.nfre,ngram从1-4，新建写入文件存入output_files 
     @return: 结果为分开的频数统计文件
     """
    """ 输出为ngram为1,2,3,4的文件，分别是一个词、两个词、三个词、四个词的频数文件,output_files是输出文件列表"""
    output_files = []
    for i in range(4) :
        output_files.append(codecs.open(input_path+'_%d'%(i+1), 'w', 'utf-8'))
    """ 读取文件"""
    input_file = codecs.open(input_path, 'r', 'utf-8')
    """ 循环文件每一行内容"""
    for line in tqdm(input_file) :
        length = len(line.split(' '))
        """ 写入对应的输出文件中"""
        output_file = output_files[length-2]
        output_file.write(line)

    for i in range(4) :
        output_files[i].flush()
        output_files[i].close()

def n_gram_unk(input_path, top_n) :
    """
     @param input_path:  n_gram_seperate处理后的结果文件
     @param top_n:  词表大小例如28k,必须包含unk词
     @return: ngram1-4按照词表处理后的频数文件，分别是xx.nfre_unk_1、xx.nfre_unk_2、
     xx.nfre_unk_3、xx.nfre_unk_4(词表外的词都用unk代替)
    """
    """ 读取1-4所有文件内容，存入datas列表中，列表每个元素是一个文件的所有内容"""
    datas = [codecs.open(input_path+'_%d'%(i+1), 'r', 'utf-8').readlines() for i in range(4)]
    deal = lambda x : [t.strip().split(' ') for t in x]
    datas = list(map(deal, datas))

    unk_word = '<UNK>'
    """ 根据传进来的参数确定词表大小"""
    words_list = [t[0] for t in datas[0][:top_n-1]] + [unk_word]
    word_set = set(words_list)

    for i, data in enumerate(datas) :
        """ 建立data_dict词表频数字典，键是词，值是频数，例如{'，': 413794,'eos1': 184953}"""
        data_dict = {}
        for dt in data :
            for j, wt in enumerate(dt[:-1]) :
                if not wt in word_set :
                    dt[j] = unk_word

            key = ' '.join(dt[:-1])
            value = int(dt[-1])
            if not key in data_dict :
                data_dict[key] = 0
            data_dict[key] += value
        """ 降序排列词表频数字典，得到降序排列的频数列表"""
        data_list = [[k, data_dict[k]] for k in data_dict]
        data_list = sorted(data_list, key=lambda x: x[1], reverse=True)
        """ 照词表处理后的频数文件，分别保存到各自文件中"""
        output = codecs.open(input_path+'_unk_%d'%(i+1), 'w', 'utf-8')
        for dt in data_list :
            output.write('%s %d\n'%(dt[0], dt[1]))
        output.flush()
        output.close()

# n_gram_seperate('trn.tgt.nfre')
# n_gram_seperate('fqtrn.tgt.nfre')
# n_gram_unk('trn.tgt.nfre', 28000)
# n_gram_unk('fqtrn.tgt.nfre', 28000)
# exit()
class Count_p() :
    """
     处理条件概率相关事项

     @return: 根据古德图灵平滑算法，返回p(b|a)=r_C(a,b)/r_C(a) 
    """

    def __load_dict(self, file_path):
        """
         @param file_path:  ngram频数文件
         @return:   ngram各自频数字典，例如三元组频数文件{'sos3 sos2 sos1': 184953, 'eos1 eos2 eos3': 184953}
        """
        lines = codecs.open(file_path, 'r', 'utf-8').readlines()
        lines = [t.strip().split(' ') for t in lines]
        return_dict = {' '.join(t[:-1]):int(t[-1]) for t in lines}
        return return_dict

    def __get_ri(self, i) :
        """
         得到频数i平滑后的数值r_i

         @param i:  频数
         @return:   频数平滑化后对应的值r_i，用来做之后的概率计算
        """
        assert type(i) == int
        dict_ri = {
            0 : 0.00027, 
            1 : 0.446, 
            2 : 1.26, 
            3 : 2.24, 
            4 : 3.24, 
            5 : 4.22, 
            6 : 5.19, 
            7 : 6.15, 
            8 : 7.10, 
            9 : 8.05, 
            10 : 9, 
            11 : 10, 
            12 : 11, 
        }
        if i in dict_ri :
            return dict_ri[i]
        else :
            return float(i - 1)

    def __init__(self, tri_gram, bi_gram) :
        """
         @param tri_gram:  三元组频数文件
         @param bi_gram:  二元组频数文件
         @return:   
        """
        self.tri_dict = self.__load_dict(tri_gram)
        self.bi_dict = self.__load_dict(bi_gram)
        print 

    def get_p(self, word_list) :
        """ 
        @param word_list:  传入的词表例如“我 爱 她”
        @return:   传入词表的概率
        """
        assert type(word_list) in [str, list]
        if type(word_list) == str :
            word_list = word_list.split(' ')
        """ 去除最后一词的字符串 """
        word_head = ' '.join(word_list[:-1])

        word_head_c = self.bi_dict.get(word_head, 0)
        if word_head_c == 0 :
            return 0.
        """ 给定参数的完整字符串 """
        word_full = ' '.join(word_list)
        word_full_c = self.tri_dict.get(word_full, 0)
        word_full_c_ri = self.__get_ri(word_full_c)
        word_head_c_ri = self.__get_ri(word_head_c)
        # print(word_full, word_full_c, word_full_c_ri)
        # print(word_head, word_head_c, word_head_c_ri)
        """ 返回输入参数的概率 """
        to_return = word_full_c_ri / word_head_c_ri
        return to_return

    def get_p_distribute(self, word_head, word_list) :
        assert type(word_head) is str
        assert type(word_list) is list
        """ 根据开头词计数，根据字典里的键值取value，如果没有这个键，默认取0 """
        word_head_c = self.bi_dict.get(word_head, 0)
        if word_head_c == 0 :
            """ 得到0分布的概率，一组0"""
            return [0.] * len(word_list)
        word_head_c_ri = self.__get_ri(word_head_c)

        word_fulls = ['%s %s'%(word_head, word) for word in word_list]
        word_full_cs = list(map(lambda x: self.tri_dict.get(x, 0), word_fulls))
        word_full_c_ris = np.array(list(map(self.__get_ri, word_full_cs)))
        """ 返回输入参数的概率 """
        to_return = word_full_c_ris / word_head_c_ri
        return to_return
        
class Decoder() :
    """
     解码语句相关

     @return: 
    """
    def __init__(self, count_p, decode_word_list, beam_size=20, max_length=50, return_topn=10, max_n_gram=3) :
        """
        @param count_p:  根据古德图灵平滑算法得到的概率值
        @param decode_word_list: 需要解码的词表
        @param beam_size: 需要保留的大概率qk的个数
        @param max_length : 序列的最大长度
        @param return_topn : 只返回topn个词序列
        @param max_n_gram : 
        """

        self.count_p = count_p
        self.beam_size = beam_size
        self.max_length = max_length
        self.return_topn = return_topn
        self.max_n_gram = max_n_gram

        self.end_judge = lambda words: all([words[-max_n_gram+i]=='eos%d'%(i+1) for i in range(max_n_gram-1)])
        
        self.decode_word_list = decode_word_list
        print('decode_word_list_len: ', len(decode_word_list))
    
    def show_list(self, list_in, verbose=0) :
        to_returns = []
        for i, t in enumerate(list_in) :
            infor_0 = 'rank_%d   score: %.4f'%(i, t['scores'])
            infor_1 = ' '.join(t['words'])
            infor = '%s\n%s\n'%(infor_0, infor_1)
            """ 打印结果列表"""
            if verbose > 0 :
                print(infor)
            to_returns.append(infor)
        return to_returns

    def decode(self, starts, max_length=None, return_topn=None, beam_size=None, pruning=True, verbose=0) :
        """
         解码语句相关
         @param starts:  两个的词语作为一句话的开始,其他参数在init中
         @return: 返回收器里面前return_topn个结果
        """
        if max_length is None :
            max_length = self.max_length
        if return_topn is None :
            return_topn = self.return_topn
        if beam_size is None :
            beam_size = self.beam_size
        start_n = starts.split(' ')
        """ 得到词序列、得分的字典列表"""
        beam = []
        beam.append({
            'words' : start_n,
            'scores' : 0,
            })
        evaluate_beam = lambda x: x['scores'] / len(x['words'])
        evaluate_threshold = lambda x: x['scores'] / max_length
        """ 创建一个回收器，存放最终的结果 """
        collector = []
        threshold = -1e20
        """ 循环总次数为序列长度"""
        all_steps = list(range(len(start_n), max_length))
        if verbose > 0 :
            all_steps = tqdm(all_steps)
        for step in all_steps :
            """ 循环终止条件是beam为空 """
            if len(beam) == 0 :
                break
            if verbose > 1 :
                print(step, len(collector))
                self.show_list(beam, verbose=1)
                print('\n')
            """ 构造一个新的beam """
            new_beam = []
            for beam_t in beam :
                """ 取出beam_t[words]中最后n个词，作为每次循环的开始词，n为HMM阶数 """
                word_head = ' '.join(beam_t['words'][-(self.max_n_gram-1):])
                """ 计算以word_head为开始，下一个词是词表中的每个词的概率，如果词表是28k,那么word_next_disb为28k个概率数组"""
                word_next_disb = self.count_p.get_p_distribute(word_head, self.decode_word_list)
                word_next_disb = list(word_next_disb)
                """ 此时说明开头词在词频统计时并不存在，说明一组概率为0"""
                if word_next_disb[0] == 0 :
                    continue
                """ 更新new_beam"""
                for i, s in enumerate(word_next_disb) :
                    new_beam.append({
                        'words'  : beam_t['words'] + [self.decode_word_list[i]],
                        'scores' : beam_t['scores'] + math.log(s),
                    })
            """ 对new_beam进行排序 """
            new_beam = LazySorted(new_beam, key=lambda x: x['scores'], reverse=True)[:beam_size]

            beam = []
            for beam_t in new_beam :
                """ 判断当前的beam_t是否有存在的必要"""
                if pruning and threshold > evaluate_threshold(beam_t) :
                    continue
                """ 判断是否是eos结束，如果是就放进回收器中，说明当前这句话解码完毕"""
                """ 如果不是，说明需要继续解码，更新beam"""
                if self.end_judge(beam_t['words']) : 
                    collector.append(beam_t)
                else :
                    beam.append(beam_t)
            """ 对每次回收器的结果进行排序，并且更新阈值"""
            if pruning and not len(collector) == 0 :
                collector = sorted(collector, key=evaluate_beam , reverse=True)
                collector = collector[:return_topn]
                threshold = evaluate_beam(collector[-1])
        """ 最后一步，需要把beam存放于回收器中"""
        if step == max_length :
            collector += beam
        """ 对回收器排序 """
        collector = sorted(collector, key=evaluate_beam , reverse=True)
        collector = collector[:return_topn]
        """ 判断是否需要打印出回收器的结果"""
        if verbose > 0 :
            self.show_list(collector, verbose=1)
        return collector
""" 得到解码前的词表频数列表"""
decode_word_list = codecs.open('trn.tgt.nfre_unk_1', 'r', 'utf-8').readlines()
""" 得到解码前的词表列表"""
decode_word_list = [t.strip().split(' ')[0] for t in decode_word_list]

# count_p3 = Count_p(tri_gram='fqtrn.tgt.nfre_unk_3', bi_gram='fqtrn.tgt.nfre_unk_2')
# count_p4 = Count_p(tri_gram='fqtrn.tgt.nfre_unk_4', bi_gram='fqtrn.tgt.nfre_unk_3')
# count_pn3 = Count_p(tri_gram='trn.tgt.nfre_unk_3', bi_gram='trn.tgt.nfre_unk_2')
# count_pn4 = Count_p(tri_gram='trn.tgt.nfre_unk_4', bi_gram='trn.tgt.nfre_unk_3')

# decoder3 = Decoder(count_p3, decode_word_list=decode_word_list)
# decoder4 = Decoder(count_p4, decode_word_list=decode_word_list, max_n_gram=4)
# decodern3 = Decoder(count_pn3, decode_word_list=decode_word_list)
# decodern4 = Decoder(count_pn4, decode_word_list=decode_word_list, max_n_gram=4)

if __name__ == '__main__':

    # count_p = Count_p(tri_gram='fqtrn.tgt.nfre_unk_3', bi_gram='fqtrn.tgt.nfre_unk_2')
    count_p = Count_p(tri_gram='trn.tgt.nfre_unk_2', bi_gram='trn.tgt.nfre_unk_1')
    print(count_p.get_p('我 的'))

    # decoder = Decoder(count_p, decode_word_list=decode_word_list)

    """ max_n_gram 需要和上面的tri_gram 、bi_gram 相对应 """
    decoder = Decoder(count_p, decode_word_list=decode_word_list, max_n_gram=2)

    """ decode的参数词语的个数需要和max_n_gram相对应 """
    collector = decoder.decode('sos1 我', verbose=1)

    # collector = decoder.pdecode('sos1 我 的', return_topn=10, verbose=1)

    