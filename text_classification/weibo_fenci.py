import csv
from langconv import *
import jieba
import jieba.posseg as pseg
from sys import argv
import codecs
import sys
import multiprocessing

def fenci(sentence):
    flag_list = ['t', 'q', 'p', 'u', 'e', 'y', 'o', 'w', 'm', 'x', 'un']
    content = sentence.strip()
    words = pseg.cut(content)
    words_list = []
    for w in words:
        flag = w.flag
        word = w.word
        if (flag not in flag_list) and (u'\u4e00' <= word[0] <= u'\u9fa5'):
            words_list.append(word)
    
    return words_list

def weibo_data_process(row):
	
	content = ''
	row = row.split(',')
	if len(row)!= 2:
		return	''	
	line = row[0]
	words_list = fenci(line)
	if len(words_list) < 2:
		return ''
	
	content = ' '.join(words_list)
	content += ' #' + row[1]
	return content

def mycallback(x):
	if(x != ''):
		with open('train_full.txt', 'a+') as f:
			f.write(x)
			

if __name__ =='__main__':

	default_encoding = 'utf-8'
	if sys.getdefaultencoding() != default_encoding:
		reload(sys)
		sys.setdefaultencoding(default_encoding)

	
	_, file_name = argv

	csv_reader = open(file_name, 'r')
	content_list = []
	
	pool = multiprocessing.Pool(multiprocessing.cpu_count())



	for line  in csv_reader:
		pool.apply_async(weibo_data_process, (line,), callback = mycallback)
	


	pool.close()
	pool.join()

	csv_reader.close()

