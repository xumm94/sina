from sys import argv
import sys
import multiprocessing



def weibo_data_process(row):
	
	content = ''
	line = row.split('\t')

	if len(line)!= 2:
		return	''	
	label = line[1]
	label = label.strip('\n')

	if label == '__label__tagCategory_050':
		return ""

	return row

def mycallback(x):
	if(x != ''):
		with open('process_without_entertainment.txt', 'a+') as f:
			f.write(x)




if __name__ =='__main__':

	default_encoding = 'utf-8'
	if sys.getdefaultencoding() != default_encoding:
		reload(sys)
		sys.setdefaultencoding(default_encoding)

	
	_, input_file_name = argv

	input_file = open(input_file_name, 'r')
	


	content_list = []


	
	pool = multiprocessing.Pool(int(multiprocessing.cpu_count() / 2))



	for line  in input_file:
		#print(line)
		pool.apply_async(weibo_data_process, (line,), callback = mycallback)

	

	pool.close()
	pool.join()

	input_file.close()


