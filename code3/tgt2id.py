# -*- coding: UTF-8 -*-
import codecs
import numpy as np
# from keras.preprocessing.sequence import pad_sequences

def special_code():
    np.random.seed(32)
    num = 100
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

def tar2id():
    a = codecs.open('../data3/voc2id', 'r', 'utf-8').readlines()
    a = [t.strip().split(' ') for t in a]
    a = {t[0]: int(t[1]) for t in a}
    tgt2id = codecs.open('../data3/dev_50.tgt.out', 'a', 'utf-8')
    devtgt = codecs.open('../data3/dev_50.tgt', 'r', 'utf-8').readlines()
    for line in devtgt:
        line = line.strip().split(' ')
        if len(line)==1:
            continue
        to_line = []
        for ele in line:
            try:
                to_line.append(a[ele])
            except:
                to_line.append(a["unk"])
        if len(to_line) < 50:
            c = 50-len(to_line)
            to_line.append(a["eos"])
            for i in range(c-1):
                to_line.append(a["msk"])
        else:
            to_line = to_line[:49]
            to_line.append(a["eos"])
        # print(to_line)
        tgt2id.write(' '.join([str(i) for i in to_line]) + '\n')


if __name__ == '__main__':
    tar2id()
    # vector_matrix()


