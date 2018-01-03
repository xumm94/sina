# coding=utf-8
import cPickle as pickle
import numpy as np
import sys


reload(sys)
sys.setdefaultencoding('utf-8')

def line_process(row, Embedding_dict):
    embedding = None
    label = None

    line = row.split('\t')
    if len(line) != 2:
        return embedding, label
    words, label_other_form = line
    words = words.split(' ')
    label = label_other_form.strip('\n').strip(' ')
    label = label.strip('__label__tagCategory_')
    label = int(label)
    embedding = np.zeros((100))

    n_word = 0
    for word in words:
        if word in Embedding_dict:
            embedding += Embedding_dict[word]
            n_word += 1

    if n_word == 0:
        return None, None

    embedding = embedding / (1.0 * n_word)


    return embedding, label


if __name__ == '__main__':

    with open("Embedding_dict.pkl", 'rb') as f:
        Embedding_dict = pickle.load(f)
    print("Loading Embedding_dict Done")

    Embedding_list = []
    label_list = []
    with open("process_no_entertain_with_other.txt", 'r') as f:
        for line in f:
            embedding, label = line_process(line, Embedding_dict)
            if embedding is not None and label is not None:
                Embedding_list.append(embedding)
                label_list.append(label)
    with open('embedding.pkl', 'wb') as f:
        pickle.dump(Embedding_list, f)

    with open('label.pkl', 'wb') as f:
        pickle.dump(label_list, f)

        








