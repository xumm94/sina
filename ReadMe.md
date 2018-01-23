 1.fasttext
  直接使用的facebook开源代码，代码github地址：https://github.com/facebookresearch/fastText
  在github中有详细的代码使用说明，我就是按照这上面的命令调用训练的。
  代码及数据放置在：10.77.9.21 /data1/shixin5/workspace/xummworkspace/xummworkspace/fastText_without_entertainment/fastText-0.1.0
  其中，有三个文件需要特殊说明：
  （1）3Ngram_3mincount_1wminlabel.bin 
  		3Ngram 表示除了jieba分词的结果，还有Ngram的一些词加入了词表之中。
  		3mincount 表示在训练集中出现次数少于3的词被忽略掉
  		1wminlabel 表示在训练集中出现次数少于1万的标签被忽略掉

  		这个是训练好的模型，可以直接在C++及python中直接使用（python中facebook是通过python调用C++的代码）
  （2）3Ngram_3mincount_1wminlabel.vec
  		这个是训练好的Embedding

  （3）process_no_entertain_with_other.txt
  		这个是去除“娱乐明星”这一类并加入其它类的训练数据，共2500万条（预处理与starspace相同，预处理文件在10.77.121.121 /data1/shixin5/xummworkspace/beifen input_data_extract.py）

  	布置在服务器上的代码放置在：10.77.6.241 /data1/shixin5/xumengmeng

  其中，有两个文件需要特殊说明：
  （1）serverw2v.py
  		这个是在部署服务器上的文件。
  （2）both_labels.pkl
  		包含所有分类类别的列表。（共41类）

  重要：在使用python调用fasttext的时候，不要使用fasttext库，要使用pyfasttext这个库。我们的模型太大，使用fasttext库无法加载模型。具体调用pyfasttext请自行查询相关文档。


 2.starspace
 	直接使用的facebook开源代码，代码github地址：https://github.com/facebookresearch/StarSpace
 	在github中有详细的代码使用说明，我就是按照这上面的命令调用训练的。
 	代码及数据放置在：10.77.9.21 /data1/shixin5/workspace/xummworkspace/xummworkspace/starspace/StarSpace
 	其中：
 	（1）2_ngrams_0.05_dropoutLHS_100_dim 
 		这个是训练好的模型，2Ngram， dropout rate 为0.05，Embedding的维度为100
 	（2）2_ngrams_0.05_dropoutLHS_100_dim.tsv
 		这个是训练好的Embedding
 	 (3) process_res.txt
 	 	 这个是训练的数据，800万条。(800万条微博分词、去除停词等)

 	 starspace训练的输入在 10.77.112.121 /data1/shixin5/xummworkspace train.csv（800万条原微博）

 3.SVM_Embedding
 	文件地址：10.77.9.21 /data1/shixin5/workspace/xummworkspace/xummworkspace/SVM_Embedding
 	利用SVM对微博进行分类。查询每条微博分词后对应的每个词的Embedding，然后取Embedding的平均。目前该模型还在跑，还没跑出来结果。
 	有几个文件需要特殊说明：
 	（1）SVM_model.py
 		SVM为二分类问题，通过这个脚本，运行得到的两两分类的模型。
 	（2）SVM_model_ovr.py
 		这个是将A类（某一类）和其他类（除了A类）分类的模型。
 	 (3) process_no_entertain_with_other.txt
 	 	经过分词、过滤后得到微博数据。
 	（4）embedding.pkl
 		59G，2500万条微博对应的Embedding，往SVM里面塞的数据。
 		此数据由weibo_Embedding_extract.py生成，该脚本通过读取微博，分词，然后利用分词查找相应的Embedding，然后对Embedding求平均，输入数据为process_no_entertain_with_other.txt
 	（5）Embedding_dict.pkl
 		字典，是中文词到Embedding的映射。由脚本Embedding.py生成，输入数据为3Ngram_3mincount_1wminlabel.vec
 	（6）SVMmodel.pkl
 		（1）中对应的模型，尚未训练出来。
 	（7）SVMmodel_ovr.pkl
 		（2）中对应的模型，尚未训练出来。	

 4.densenet
 	代码、数据以及相应的使用文档均在 10.77.9.21 /data1/shixin5/workspace/xummworkspace/xummworkspace/densenet

 5.Voice-gender
 	服务器上原有文件已经丢失，在自己的电脑上找了一下，将相应的代码上传到了 10.77.9.21 /data1/shixin5/workspace/xummworkspace/xummworkspace/voicegender
 	其中，几个文件需要说明
 	（1）feature_extract.R
 		R语言编写的程序，用于提取声音的特征。
 		R程序相关以及数据集下载请参考https://github.com/primaryobjects/voice-gender
 	（2）load_model_test.py
 		用于线下测试python调用R提取特征，然后用xgboost分类。
 	（3）voiceGender.py
 		可以部署在服务器上运行。

 	说明：在运行voiceGender.py之前需要预装的环境：
 	（1）R
 	（2）xgboost（python库）
 	（3）rpy2（python库）





