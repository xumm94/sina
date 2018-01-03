# coding=utf-8
import cPickle as pickle
import numpy as np
import sys


reload(sys)
sys.setdefaultencoding('utf-8')


if __name__ == '__main__':
	Embedding_dict = dict()
	with open('3Ngram_3mincount_1wminlabel.vec') as f:
    	    i = 0
    	    for line in f:
        	if i == 0:
            	    i += 1
            	    continue
        
        	line_split = line.strip('\n').strip(' ').split(' ')
        	Embedding_dict[line_split[0]] = np.array([float(num) for num in line_split[1:]])
        	i += 1

	with open('Embedding_dict.pkl', 'wb') as f:
    	    pickle.dump(Embedding_dict, f)





