# coding=utf8
# -*- coding: UTF-8 -*-
'''
这个python脚本用来产生nmt所需的训练（测试）语料和单词。
输入为一个大的文本文件
处理流程为：分词 --> 得到单词表 --> 将分词后的文本处理成一行一行  -->  产生src,tgt训练语料对
'''

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import jieba
import codecs
from collections import Counter
import re


def tokenizer_file(file_path, des_path):
    '''
    使用结巴分词
    :param file_path: 输入文件，纯文本格式，utf-8编码
    :param des_path: 分词后的文件
    :return:
    '''
    f = codecs.open(file_path, 'r', 'utf-8')
    text = f.read()
    f.close()
    toked_text = ' '.join(jieba.cut(text))
    f = codecs.open(des_path, 'w', 'utf-8')
    f.write(toked_text)
    f.close()


def get_vocab(file_path, vocab_path, vocab_count):
    '''
    从文件中抽取字典，将出现频率前n的词放入字典文件 ，n由vocab_count来指定
    需要特别注意的是：在NMT模型中，字典的前三行要求是<unk><s></s>,可以手动输入
    :param file_path: 文件路径，要求已经分好词，utf-8格式
    :param vocab_path: 词典路径
    :param vocab_count: 词典大小
    :return:
    '''
    lines = codecs.open(file_path, 'r', 'utf-8').readlines()
    c = Counter()
    for line in lines:
        word_list = line.split()
        for e in word_list:
            c[e] += 1
    most_word = c.most_common(vocab_count)
    word_id_list = [e[0] for e in most_word]
    vf = codecs.open(vocab_path, 'w', 'utf-8')
    vf.write('<unk>\n<s>\n</s>\n')  # 字典的前三行是依次<unk><s></s>，nmt模型需要
    for e in word_id_list:
        vf.write(e)
        vf.write('\n')
    vf.close()


def load_vocab(vocab_path):
    '''
    将上一个函数产生的字典load进内存
    :param vocab_path: 字典路径
    :return: 字典以list的形式返回
    '''
    vocab = []
    lines = codecs.open(vocab_path, 'r', 'utf-8').readlines()
    for line in lines:
        vocab.append(line.strip())
    return vocab


def toked2lines(tokened_data_path, line_file_path):
    '''
    将分词好的文章切分成一句一句，以，。？！：这些标点符号作为切分依据，但保留标点符号。例如对“我是中国人，我爱我的祖国。”
    这句话将切分为：
    我是中国人，
    我爱我的祖国。
    需要注意的是，切分的时候也将句子中的双引号删掉了，主要考虑是保留双引号的时候切分完之后会出现”开头的句子。
    :param tokened_data_path:分词好的文本路径
    :param line_file_path: 切分好后的路径
    :return:
    '''

    def conver2list(toked_paragraph):
        rs_list = []
        stop_token = set([u'，', u'。', u'？', u'！', '：'])
        token_list = toked_paragraph.split()
        start_index = 0
        for i in range(len(token_list)):
            if token_list[i] in stop_token:
                rs_list.append(' '.join(token_list[start_index:i + 1]))
                start_index = i + 1
        if start_index < len(token_list):
            rs_list.append(' '.join(token_list[start_index:]))
        return rs_list

    toked_data = codecs.open(tokened_data_path, 'r', 'utf-8')
    df = codecs.open(line_file_path, 'w', 'utf-8')
    for paragraph in toked_data:
        paragraph = paragraph.replace('“', '')
        paragraph = paragraph.replace('”', '')
        sentence_list = conver2list(paragraph)
        if len(sentence_list) < 2:
            continue
        for line in sentence_list:
            df.write(line + '\n')
        df.write('<eos>' + '\n')
    df.close()


# def generate_nmt_data(tokened_data_path, src_path, des_path):
#     def split_helper(paragraph):
#         end_token = '。.?？！!;；'
#         end_token_patter = '[。.?？！!;；]'
#         rs_list = re.split(end_token_patter, paragraph)
#         return rs_list
#
#     raw_data = codecs.open(tokened_data_path, 'r', 'utf-8')
#     sf = codecs.open(src_path, 'w', 'utf-8')
#     df = codecs.open(des_path, 'w', 'utf-8')
#     for line in raw_data:
#         line = line.strip('\n\r\t ')
#         if len(line) == 0:
#             continue
#         line = line.replace('“', '')
#         line = line.replace('”', '')
#         sentences = split_helper(line)
#         for i in range(len(sentences) - 1):
#             sf.write(sentences[i] + '\n')
#             df.write(sentences[i + 1] + '\n')
#         sf.write(sentences[-1] + '\n')
#         df.write('<eos>' + '\n')
#     sf.close()
#     df.close()


def generate_nmt_data_from_lines(lines_path, source_path, target_path, source_count):
    '''
    将toked2lines函数产生的一行一行的文本处理成NMT模型需要的样子。假如入 a,b,c,d,e,f,<eos>是一段话，其中a,b,c...f均是一句话，
    <eos>表示一段话结束，训练语料的组织方式如下所示：
    src     tgt
    a       b
    b       c
    c       d
    d       e
    e       f
    f       <eos>
    即前面一句作为源语言，后面一句作为目标语言。当然直接这么做容易出现主题飘移。为了减缓主题漂移，可以用前两句或者三句作为目标语言，如：
    src     tgt
    a,b     c
    b,c     d
    c,d     e
    d,e     f
    e,f     <eos>
    :param lines_path:
    :param source_path:
    :param target_path:
    :param source_count:用前多少句作为源语言，例如上面的例子中用前两句作为源语言，source_count=2
    :return:
    '''
    lines = codecs.open(lines_path, 'r', 'utf-8').readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n\r ')
    sf = codecs.open(source_path, 'w', 'utf-8')
    df = codecs.open(target_path, 'w', 'utf-8')
    for i in range(1, len(lines)):
        tgt = lines[i]
        start_index = i - source_count if (i - source_count + 1) > 0 else 0
        src = ' '.join(lines[start_index:i])
        sf.write(src + '\n')
        df.write(tgt + '\n')
    sf.close()
    df.close()


# tokenizer->getvocab->toked2lines->generate_nmt_data_from_lines
tokenizer_file('zhiku_data/zhiku_all_raw', 'zhiku_data/zhiku_all_toked')
get_vocab('zhiku_data/zhiku_all_toked', 'zhiku_data/zhiku_vocal_60000', 60000)
toked2lines('zhiku_data/zhiku_all_toked', 'zhiku_data/zhiku_all_toked_lines')
generate_nmt_data_from_lines('zhiku_data/zhiku_all_toked_lines', 'zhiku_data/src_file', 'zhiku_data/tgt_file', 2)
