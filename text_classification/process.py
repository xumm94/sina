# coding=utf-8
import os
import sys
import re
import jieba
import json
from tqdm import tqdm


reload(sys)
sys.setdefaultencoding('utf-8')


class ProcessText():
	def __init__(self):
		self.zh_pattern_re = re.compile(u'[\u4e00-\u9fa5]+')
		self.emotion = re.compile(r'\[(.*?)\]')

		jieba.load_userdict('data/stopwords.txt')
		self.stopwords_set = set()
		self.enkeywords_set = set()

		for line in open('data/stopwords.txt'):
			self.stopwords_set.add(line.decode('utf-8').strip())
		for line in open('data/en_keywords.txt'):
			self.enkeywords_set.add(line.decode('utf-8').strip())
	
	def process(self, content):  
		flag = True					  
		term_list = jieba.lcut(content)
		newterm_list = [term for term in term_list if term not in self.stopwords_set] # 过滤掉停用词 
		only_chn = []
		for linn in newterm_list:
			if linn == u'[':  #去除[]内容
				flag = False	
			if linn == u']':
				flag = True
			if flag:
				if linn in self.enkeywords_set:
					only_chn.append(linn)
				else:
					match = self.zh_pattern_re.search(linn)
					if match:
						only_chn.append(match.group())
		return ' '.join(only_chn)

if __name__ == '__main__':
	processtext = ProcessText()
	res_file = open('res/process_res.txt','w')
	weibo_file = open('res/yuan_weibo.txt','w')
	for line in tqdm(open('data/remain2recall.3days.13')):
		#mid, tag1, content = line.strip().split('\t')
		weibo_data = line.strip().split('\t')
		if(len(weibo_data) != 2):
			continue
		else:
			mid = weibo_data[0]
			content = weibo_data[1]
		#tag = tag1.split(':')[1].split('@')[0]
		remain_text = processtext.process(content)
		#new_line = mid + '\t' + tag + '\t' + remain_text + '\n'
		#new_line = remain_text + '\t__label__' + tag + '\n'
		new_line = remain_text + '\n'
		res_file.write(new_line)
		weibo_file.write(line)
	res_file.close()
	weibo_file.close()







