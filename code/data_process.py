# coding=utf8
# -*- coding: UTF-8 -*-
'''
处理流程为：分词 --> 得到单词表 --> 将分词后的文本处理成一行一行  -->  产生src,tgt训练语料对
src 每行100个词，tgt 每行为src的后面的一句话， 100作为参数，可以修改
'''
import jieba
import codecs
from collections import Counter
import re,os

def fenchapter(file_path, des_path):
    f = codecs.open(file_path, 'r', 'utf-8')
    text = f.readlines()
    f.close()
    chapter=""
    i=1
    for line in text:
        line = line.strip()
        if len(line)==0:
           if len(chapter)!=0:
               fw = codecs.open(des_path+'/'+str(i)+'.txt','w','utf-8')
               fw.write(chapter)
               fw.close()
               i=i+1
           chapter=""
        else:
            chapter+=line
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
def get_src_tar(file_path, src_path, tar_path, vocab_size=100):
    '''
    :param file_path: 需要切分的文件，要求是已经分好词的文件
    :param src_path: 源文件路径
    :param tar_path: 目标文件路径
    :param vocab_size: 切分大小，default为100，每100个词为一行
    :return:
    '''
    lines = codecs.open(file_path, 'r', 'utf-8').readlines()
    src_file = codecs.open(src_path, 'w', 'utf-8')
    tar_file = codecs.open(tar_path, 'w', 'utf-8')
    word_list=[]
    for line in lines:
        word_list += line.strip().split(' ')

    # for e in word_list:
    #     print(e)
    # exit()
    src_line, tar_line, src_lines, tar_lines, tar_last, src_last= [], [], [], [], [], []
    flag = 0
    # flag是0写入src flag是1写入tar
    first_line = True
    t0 = []
    for i in range(len(word_list)):
        e = word_list[i]
        if first_line:
            src_line.append(e)
            if len(src_line) < vocab_size or src_line[-1] not in [u'。', u'？', u'！']:
                continue
            else:
                src_line = src_line[-vocab_size:]
                src_lines.append(' '.join(src_line))
                src_last = src_line
                src_line = []
                flag = 1
                first_line = False
                continue
        if flag == 0 and not first_line:
            src_line = src_last + tar_last
            src_line = src_line[-vocab_size:]
            src_lines.append(' '.join(src_line))
            src_last = src_line
            t0.append(e)
            src_line = []
            flag = 1
            continue
        if flag == 1 and e not in [u'。', u'？', u'！']:
            if len(t0)!=0:
                tar_line.append(t0[0])
                t0=[]
            tar_line.append(e)
            # if e not in [u'。', u'？', u'！']:
            continue
        elif e in [u'。', u'？', u'！']:
            tar_line.append(e)
            tar_last = tar_line
            tar_lines.append(' '.join(tar_line))
            tar_line = []
            flag = 0

    if len(tar_last) < vocab_size:
        count = vocab_size - len(tar_last)
        src_line = src_last[-count:]+tar_last
    else:
        src_line = tar_last[-vocab_size:]
    src_lines.append(' '.join(src_line))
    tar_lines.append('<EOF>')
    for i in range(len(src_lines)-len(tar_lines)) :
        src_lines.pop()
    src_file.write('\n'.join(src_lines))
    tar_file.write('\n'.join(tar_lines))
    src_file.flush()
    tar_file.flush()
    src_file.close()
    tar_file.close()




if  __name__ == '__main__':
    ''' step1 分词、分章节 '''
    '''
    files = os.listdir('../data/我吃西红柿')
    for f in files:
    #     isExists = os.path.exists('../data/我吃西红柿_token/'+f.split('.')[0])
    #     if not isExists:
    #         os.makedirs('../data/sanlianban_token/'+f.split('.')[0])
    #         os.makedirs('../data/sanlianban_src/' + f.split('.')[0])
    #         os.makedirs('../data/sanlianban_tar/' + f.split('.')[0])
        tokenizer_file('../data/我吃西红柿/'+ f , '../data/我吃西红柿token/'+f)
    # + f.split('.')[0]+'/'
    #     fenchapter('../data/sanlianban_token/'+f.split('.')[0]+'/'+f, '../data/sanlianban_token/'+f.split('.')[0])
    exit()
    '''
    ''' step2 生成每个章节的src、tar '''
    files = os.listdir('../data/我吃西红柿token/2/')
    for f in files:
        get_src_tar('../data/我吃西红柿token/2/' + f,
                    '../data/我吃西红柿_src/' + f,
                    '../data/我吃西红柿_tar/' + f, 100)
    exit()
        # print(f)
        # f = f.split('.')[0]
        # count = len(os.listdir('../data/sanlianban_token/'+f))
        # for i in range(1, count):
        #     get_src_tar('../data/sanlianban_token/'+f+'/'+str(i)+'.txt', '../data/sanlianban_src/'+f+'/'+str(i)+'.txt',
        #             '../data/sanlianban_tar/'+f+'/'+str(i)+'.txt',100)

    ''' 单独处理不分章节的文章 '''
    # get_src_tar('../data/sanlianban_token/越女剑/越女剑.TXT',
    #                 '../data/sanlianban_src/越女剑/越女剑.txt',
    #                 '../data/sanlianban_tar/越女剑/越女剑.txt', 100)
    # get_src_tar('../data/sanlianban_token/1.txt',
    #                 '../data/sanlianban_src/1.txt',
    #                 '../data/sanlianban_tar/1.txt', 3)
    ''' step3 将每个小说的每个章节整合得到每个小说总的src '''
    # files = os.listdir('../data/sanlianban_src')
    # for f in files:
    #     src_content = []
    #     count = len(os.listdir('../data/sanlianban_src/' + f))
    #     for i in range(1,count):
    #         fd = codecs.open('../data/sanlianban_src/' + f + '/' + str(i) + '.txt', 'r', 'utf-8')
    #         text = fd.read().strip()
    #         fd.close()
    #         src_content.append(text)
    #     fw = codecs.open('../data/sanlianban_src/' + f + '/' + f + '.txt','w','utf-8')
    #     fw.write('\n'.join(src_content))
    #     fw.close()

    ''' step4 将每个小说的每个章节整合得到每个小说总的tar '''
    # files = os.listdir('../data/sanlianban_tar')
    # for f in files:
    #     tar_content = []
    #     count = len(os.listdir('../data/sanlianban_tar/' + f))
    #     for i in range(1,count):
    #         fd = codecs.open('../data/sanlianban_tar/' + f + '/' + str(i) + '.txt', 'r', 'utf-8')
    #         text = fd.read()
    #         fd.close()
    #         tar_content.append(text)
    #     fw = codecs.open('../data/sanlianban_tar/' + f + '/' + f + '.txt','w','utf-8')
    #     fw.write('\n'.join(tar_content))
    #     fw.close()