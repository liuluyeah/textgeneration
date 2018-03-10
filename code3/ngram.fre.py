# -*- coding: UTF-8 -*-
import codecs

def ngram():
    fd = codecs.open('../data4/fqtrn.tgt', 'r', 'utf8')
    fw = codecs.open('../data4/fqtrn.tgt.sos.eos', 'a', 'utf8')
    for line in fd.readlines():
        if len(line.strip()) > 1:
            fw.write('sos3 sos2 sos1 ' + line.strip() + ' eos1 eos2 eos3'+ '\n')
    fw.close()
    fd.close()

def nfre():
    fd = codecs.open('../data4/fqtrn.tgt.sos.eos', 'r', 'utf8')
    content = []
    for line in fd.readlines():
        content += line.strip().split(' ')
    fd.close()
    voc1={}
    for ele in content:
        if ele in voc1:
            voc1[ele] += 1
        else:
            voc1[ele] = 1
    content2 = [content[i]+' ' + content[i+1] for i in range(len(content)-1)]
    voc2={}
    for ele in content2:
        if ele in voc2:
            voc2[ele] += 1
        else:
            voc2[ele] = 1
    content3 = [content[i]+' ' + content[i+1] +' '+ content[i+2]  for i in range(len(content)-2)]
    voc3={}
    for ele in content3:
        if ele in voc3:
            voc3[ele] += 1
        else:
            voc3[ele] = 1
    content4 = [content[i]+' ' + content[i+1] +' '+ content[i+2]+' '+ content[i+3]  for i in range(len(content)-3)]
    voc4={}
    for ele in content4:
        if ele in voc4:
            voc4[ele] += 1
        else:
            voc4[ele] = 1

    voc1.update(voc2)
    voc1.update(voc3)
    voc1.update(voc4)
    fw = codecs.open('../data4/fqtrn.tgt.nfre', 'a', 'utf8')
    for k,v in sorted(voc1.items(), key=lambda x:x[1], reverse=True):
        fw.write(k + ' ' + str(v) + '\n')
    fw.close()

if __name__ == '__main__':
    ngram()
    nfre()
