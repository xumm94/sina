import multiprocessing
import sys

def weibo_count(lock,row, weibo_dict):
    line = row.split('\t')
    #print(len(line))
    
    if len(line) != 2:
        return
    label = line[1].strip('\n')
    label = label.strip('__label__')
    #print(label)
    with lock:
        weibo_dict.setdefault(label, 0)
        weibo_dict[label] += 1
    
    return 
    



if __name__ =='__main__':
    
    default_encoding = 'utf-8'
    if sys.getdefaultencoding() != default_encoding:
        reload(sys)
        sys.setdefaultencoding(default_encoding)

    manager = multiprocessing.Manager()
    lock = manager.Lock()
    weibo_dict = manager.dict()
    pool = multiprocessing.Pool(int(multiprocessing.cpu_count()) / 2 )
    input_file = open('process_without_entertainment.txt', 'r')

    #print('doubi')
    for line in input_file:
        pool.apply_async(weibo_count, (lock, line,weibo_dict))

    
    pool.close()
    pool.join()
    input_file.close()

    final_result = {}

    with open('20171220_process_without_entertainment_tongji.txt', 'w') as f:
        for key, value in weibo_dict.items():
            content = key.strip('\n') + ":" + str(value) + "\n"
            f.writelines(content)




