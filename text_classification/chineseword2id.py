import pickle
import numpy as np
import os


def line_process(line, words2id, label2id):
    label, content = line.split('\t')

    label_id = label2id[label]
    words_list = content.split(' ')

    words_id_list = []

    for word in words_list:
        if word in words2id:
            words_id_list.append(words2id[word])

    if len(words_id_list) == 0:
        words_id_list = None

    return label_id, words_id_list

if __name__ == '__main__':

    with open("word2id.pkl", 'rb') as f:
        word2id = pickle.load(f)

    with open("category2id.pkl", 'rb') as f:
        label2id = pickle.load(f)

    data_dir = 'cnews_fenci'

    for filename in os.listdir(data_dir):

        pre, _ = filename.split('.')

        articles = []
        labels = []

        file = os.path.join(data_dir, filename)
        with open(file, 'r', encoding= 'utf-8') as f:
            for row in f:
                label_id, words_list = line_process(row, word2id, label2id)
                if len(words_list) < 10 :
                    continue

                articles.append(words_list)
                labels.append(label_id)

        output_dir = "cnews_chinese2id"
        output_article = pre + "_article_content.pkl"
        output_label = pre + "_article_label.pkl"

        output_article_file = os.path.join(output_dir, output_article)
        output_label_file = os.path.join(output_dir, output_label)
        with open(output_article_file, 'wb') as f:
            pickle.dump(np.array(articles), f)

        with open(output_label_file, 'wb') as f:
            pickle.dump(np.array(labels), f)