#/usr/bin/python
#coding=utf-8
import os,sys
import zipfile
from unrar import rarfile
import jieba,codecs
import chardet

def detect_walk(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            # if filename.endswith('.zip'):
            #     print(filename)
            #     zpfd = zipfile.ZipFile(root+'\\'+filename)  # 读取压缩文件
            #     os.chdir(r'E:\sunbo\武侠玄幻合集\武侠玄幻合集\武侠全集\decompress')  # 转到存储路径
            #     zpfd.extractall()
            #     zpfd.close()
            if filename.endswith('.rar'):
                print(filename)
                file = rarfile.RarFile(root+'\\'+filename)
                os.chdir(r'E:\sunbo\武侠玄幻合集\武侠玄幻合集\武侠全集\decompress')  # 转到存储路径
                file.extractall('you_want_path')  # 这里写入的是你想要解压到的文件夹
def detect_file(dir_path):
    count=0
    fw = codecs.open(r'E:\sunbo\武侠玄幻合集\武侠玄幻合集\wuxia.text', 'a', 'utf-8')
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith('.txt') or filename.endswith('.TXT'):
                try:
                    f = codecs.open(root + '\\' + filename, 'r','utf-8')
                    text = f.readlines()
                    f.close()
                except:
                    try:
                        f = codecs.open(root + '\\' + filename, 'r', 'gbk')
                        text = f.readlines()
                        f.close()
                    except:
                        try:
                            f = codecs.open(root + '\\' + filename, 'r', 'ascii')
                            text = f.readlines()
                            f.close()
                        except:
                            try:
                                f = codecs.open(root + '\\' + filename, 'r', encoding='gb2312')
                                text = f.readlines()
                                f.close()
                            except:
                                try:
                                    f = codecs.open(root + '\\' + filename, 'r', encoding='utf-16')
                                    text = f.readlines()
                                    f.close()
                                except:
                                    continue
                                    # count+=1
                                    # print(root + '\\' + filename)

                for line in text:
                    word_list = list(jieba.cut(line.strip()))
                    for e in word_list:
                        if e in [u'。', u'？', u'！', u'，','：',u'《',u'》',u'“',u'、',u'”']:
                            word_list.remove(e)
                    toked_text = ' '.join(word_list)
                    fw.write(toked_text + '\n')
    fw.close()
    print(count)

def to_utf8(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith('.txt') or filename.endswith('.TXT'):
                f = codecs.open(root + '\\'+ filename, 'r')
                ff = f.read()
                print(ff)
                exit()
                file_object = codecs.open(root + '\\' + filename, 'w', 'utf-8')
                file_object.write(ff)


if __name__ == '__main__':
    # detect_walk(r'E:\sunbo\武侠玄幻合集\武侠玄幻合集\武侠全集')
    detect_file(r'E:\sunbo\武侠玄幻合集\武侠玄幻合集\武侠全集')
    # to_utf8(r'E:\sunbo\武侠玄幻合集\武侠玄幻合集\武侠全集')
    # f=codecs.open(r'E:\sunbo\武侠玄幻合集\武侠玄幻合集\武侠全集\《金庸小说全集 14部》全（TXT）作者：金庸\《金庸小说全集 14部》全（TXT）作者：金庸\倚天屠龙记\YITIAN01.TXT','r')
    # text=f.read()
    # f.close()
    # print(text)