import os
import jieba
import jieba.posseg as pseg
import pickle
import multiprocessing


def fenci(line, words_dict):
    flag_list = ['t', 'q', 'p', 'u', 'e', 'y', 'o', 'w', 'm', 'x', 'un']
    label, content = line.strip().split('\t')
    words = pseg.cut(content)
    words_list = []
    for w in words:
        flag = w.flag
        word = w.word
        if (flag not in flag_list) and (u'\u4e00' <= word[0] <= u'\u9fa5'):
            words_list.append(word)
            words_dict.setdefault(word, 0)
            words_dict[word] += 1
    new_content = ' '.join(words_list )
    new_line = '\t'.join([label, new_content])
    return  new_line

if __name__ == '__main__':

    words_dict = {}
    file_dir = 'cnews'
    for file_name in os.listdir(file_dir):

        file = os.path.join(file_dir, file_name)
        output_file = open(file_name + '_fenci.txt', 'a', encoding= 'utf-8')

        #print(file)


        with open(file, 'r', encoding= 'utf-8') as f:
           # i = 0

            for line in f:
                new_line = fenci(line, words_dict) + '\n'
                output_file.write(new_line)
               # i += 1
               # if i % 100 == 0:
               #     print(i)
                    
        output_file.close()

