# -*- coding: UTF-8 -*-
import codecs
import numpy as np

def special_code():
    np.random.seed(32)
    unk = np.random.randn(num) / 3
    sos = np.random.randn(num) / 3
    eos = np.random.randn(num) / 3
    msk = np.zeros(100)
    a = ' '.join(['%.6f'%(t) for t in unk])
    print(a)

def voc_id():
    fd = codecs.open('../data/100.vector', 'r' , 'utf-8')
    fw = codecs.open('../data/voc2id', 'a' , 'utf-8')
    fw.write('msk 0' + '\n')
    fw.write('unk 1' + '\n')
    fw.write('sos 2' + '\n')
    fw.write('eos 3' + '\n')
    content = fd.readlines()
    for i in range(1, 39997):
        fw.write(content[i].split(' ')[0] + ' ' + str(i+3) + '\n')
    fw.close()
    fd.close()

def vector_matrix():
    np.random.seed(32)
    num = 100
    unk = np.random.randn(num) / 3
    sos = np.random.randn(num) / 3
    eos = np.random.randn(num) / 3
    msk = np.zeros(100)
    fd = codecs.open('../data/100.vector', 'r', 'utf-8')
    content = fd.readlines()
    fw = codecs.open('../data/vector.matrix.100','a','utf8')
    fw.write(' '.join(['%.6f'%(t) for t in msk]) + '\n')
    fw.write(' '.join(['%.6f' % (t) for t in unk]) + '\n')
    fw.write(' '.join(['%.6f' % (t) for t in sos]) + '\n')
    fw.write(' '.join(['%.6f' % (t) for t in eos]) + '\n')
    for i in range(1, 39997):
        fw.write(' '.join(content[i].split(' ')[1:]) + '\n')
    fw.close()
    fd.close()

if __name__ == '__main__':
    vector_matrix()


