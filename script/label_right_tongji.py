from sys import argv
import sys
import cPickle as pickle
from content_test import ContCmp






if __name__ =='__main__':

	default_encoding = 'utf-8'
	if sys.getdefaultencoding() != default_encoding:
		reload(sys)
		sys.setdefaultencoding(default_encoding)

	
	with open("both_labels.pkl", "rb") as f:
		labels_list = pickle.load(f)

	contcmp = ContCmp("root_feature_file.allid")

	weibo_all_num = 0
	weibo_pre_right = 0

	with open('20171225_data.txt', 'r') as f:
		weibo_content = f.readlines()

	with open('20171225_data_prediction_result.txt', 'r') as f:
		weibo_label = f.readlines()

	with open('English2Chinese.pkl', 'rb') as f:
		English2Chinese = pickle.load(f)

	with open("yuan_weibo.txt", 'rb') as f:
		origin_weibo = f.readlines()

	f = open('weibo_label_prob_result.txt', 'w')

	for i in range(len(weibo_content)):
		weibo_all_num += 1
		label, prob = weibo_label[i].strip().split(' ')
		label = label.strip('__label__')
		label = '1042015:' + label
		prob = float(prob)
		if label not in labels_list or prob < 0.8:
			continue

		words_list = weibo_content[i].strip().split(' ')
		content = ''.join(words_list)
		flag, _ = contcmp.check_is_exist(label, content)
		if flag :
			weibo_pre_right += 1
			if label in English2Chinese:
				label = English2Chinese[label] + '@' + str(prob)
				_, content = origin_weibo[i].split('\t')
				weibo = label +'\t' +  content
				f.write(weibo)

	print("weibo_all_num :", weibo_all_num)
	print("weibo_pre_right:", weibo_pre_right)
	print("Acc:", 1.0 * weibo_pre_right / weibo_all_num)
	f.close()








