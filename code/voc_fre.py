# -*- coding: UTF-8 -*-
import codecs
import os
import pickle


if __name__ == '__main__':
    files = os.listdir('../data/我吃西红柿token/2')
    word_dict = {}
    for f in files:
        word_lst = []
        fr = codecs.open('../data/我吃西红柿token/2/'+f,'r','utf-8')
        word_lst = fr.read().split(' ')
        for item in word_lst:
            item = item.strip()
            if item not in [u'。', u'？', u'！', u'，','：',u'《',u'》',u'“',u'、',u'”',
                            u'…', u'（', u'）'] and item!='':
                if item not in word_dict:
                    word_dict[item] = 1
                else:
                    word_dict[item] += 1
    output = open('../data/pickle_dict/我吃西红柿.pk', 'wb')
    pickle.dump(word_dict, output)
    output.close()