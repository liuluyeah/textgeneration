/home/liulu/文本生成/data4:

>fq.in40k.out28k   番茄小说in、out结果文件(in、out词表大小分别是400000,28000)

>fq.in40k.out40k   番茄小说in、out结果文件(in、out词表大小分别是400000,40000)

>jy.in40k.out28k   金庸小说in、out结果文件(in、out词表大小分别是400000,28000)

>jy.in40k.out40k   金庸小说in、out结果文件(in、out词表大小分别是400000,40000)

.in文件、.out文件说明：

>根据词语id映射文件得到的每行为词语id的文件,in每行以sos词语对应的id作为开头，out每行以eos词语对应的id作为结尾，每行固定为50个字符，不足在最末尾用msk对应的id补齐
					   
>.in文件每行形如： “2 7 236 1111 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0”

>.out文件每行形如：“7 253 8 27 148 1 2389 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0”


textgeneration/code:
>data_process.py  预处理文件，输入原始文件，得到src、tar结果，存放于data目录中：

>voc2id.py        将词表转为id ，结果voc2id28k、voc2id40k

>voc_fre.py       原始文件分词后的词频统计，结果是voc词频统计



textgeneration/code4:

>fqtgt2id.py   将分词文件转为词id文件，结果存放于data4中，

>ngram.fre.py  ngram频数统计，结果存放于data4中，分别是:fqtrn.tgt.nfre、trn.tgt.nfre


textgeneration/decode:

>n_gram_decode_copy.py  HMM解码代码，原理详见注释
其它为n_gram_decode_copy.py执行过程中产生的结果



