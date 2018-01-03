#!/usr/bin/env python
# encoding: utf-8

import sys
from TrieTree import *

class ContCmp:
	def __init__(self, lev123_cat_abi_info):
		self.dic = {}
		self.dic = self.load_all_dic(lev123_cat_abi_info)
	
	def load_all_dic(self, lev123_cat_abi_info):
		dic = {}
		for line in file(lev123_cat_abi_info):
			vec = line.strip().split('\t')
			dic.setdefault(vec[0], Trie())
			dic[vec[0]].insert(vec[1])
		return dic
	
	def check_is_exist(self, lev1id, content):
		flag = False
		ret = self.dic[lev1id].searchphrase(content)
		if len(ret.keys()) > 0: flag = True
		return flag, len(ret.keys())

if __name__ == '__main__':
	#contcmp = ContCmp(sys.argv[1])
	contcmp = ContCmp('root_feature_file.allid')
	print contcmp.check_is_exist('1042015:tagCategory_020', '猩猩惹毛了狗狗到怀疑狗生')
