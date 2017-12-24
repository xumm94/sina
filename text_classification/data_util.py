import os
import pickle
import numpy as np


def load_data():

    with open('cnews_chinese2id/cnews_train_article_content.pkl', 'rb') as f:
        train_x = pickle.load(f)

    with open('cnews_chinese2id/cnews_val_article_content.pkl', 'rb') as f:
        val_x = pickle.load(f)

    with open('cnews_chinese2id/cnews_test_article_content.pkl', 'rb') as f:
        test_x = pickle.load(f)

    with open('cnews_chinese2id/cnews_train_article_label.pkl', 'rb') as f:
        train_y = pickle.load(f)

    with open('cnews_chinese2id/cnews_val_article_label.pkl', 'rb') as f:
        val_y = pickle.load(f)

    with open('cnews_chinese2id/cnews_test_article_label.pkl', 'rb') as f:
        test_y = pickle.load(f)

    with open('word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)
        vocab_size = len(word2id.keys())

    index = np.arange(len(train_x))
    np.random.shuffle(index)

    train_x = train_x[index]
    train_y = train_y[index]

    return train_x, train_y, val_x, val_y, test_x, test_y, vocab_size
