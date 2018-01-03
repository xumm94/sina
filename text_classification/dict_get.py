#!/usr/bin/env python
#coding=utf-8
import cPickle as pickle


if __name__ =='__main__':

	tag_dict = {}	

	input_file = open('category_ability_info', 'r')

	for line in input_file:
		columns = line.split('\t')
		tag_num = columns[1]
		tag_chi = columns[0]

		if tag_num not in tag_dict:
			tag_dict[tag_num] = tag_chi

	print(len(tag_dict.keys()))
	pickle.dump(tag_dict, open('tag_dict.pkl', 'wb'))
	

	input_file.close()






