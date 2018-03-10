#/usr/bin/python
#coding=utf-8
import sys,chardet,os
import codecs

def filter( srcFile, encode, trgFile ):
	error = True
	while error:
		try:
			fs = open( srcFile, encoding = encode )
			content = fs.read()
			fs.close()
			error = False
			print( 'Success!' )
		except UnicodeDecodeError as ude:
			pos1, pos2 = ude.args[2], ude.args[3]
			fs = open( srcFile, 'rb' )
			content = fs.read()
			fs.close()
			content = content[:pos1] + content[pos2+1:]
			ofs = open( srcFile, 'wb' )
			ofs.write( content )
			ofs.close()

		except:
			print( 'Other error happened!' )
			break

def detect_file(dir_path):
	for root, dirs, files in os.walk(dir_path):
		for filename in files:
			if filename.endswith('.txt') or filename.endswith('.TXT'):
				f = codecs.open(root + '\\' + filename, 'r')
				try:
					f.read()
					f.close()
				except:
					f.close()
					tt = open(root + '\\' + filename, 'rb')
					ff = tt.read()
					enc = chardet.detect(ff)
					print(enc['encoding'])
					filter(root + '\\' + filename, enc['encoding'], root + '\\' + filename)




if __name__ == '__main__':
	detect_file(r'E:\sunbo\武侠玄幻合集\武侠玄幻合集\武侠全集\《黄易作品集 39部》全（TXT）作者：黄易\《黄易作品集 39部》全（TXT）作者：黄易')
	# filter(r'E:\sunbo\武侠玄幻合集\武侠玄幻合集\武侠全集\decompress\you_want_path\《花心帅哥大(独步香尘)》作者：颜斗.txt',
	# 	   'gbk',
	# 	   r'E:\sunbo\武侠玄幻合集\武侠玄幻合集\武侠全集\decompress\you_want_path\《花心帅哥大(独步香尘)》作者：颜斗.txt')
	# if len(sys.argv) < 4:
	# 	print( 'Usage:' )
	# 	print( sys.argv[0] + ' input-file input-encoding output-file' )
	# else:
	# 	filter( sys.argv[1], sys.argv[2], sys.argv[3] )

