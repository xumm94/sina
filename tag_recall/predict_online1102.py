#!/usr/bin/env python
# encoding: utf-8

import sys, time
from scipy import *
import scipy
from numpy import *

from sklearn import preprocessing
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from langconv import *
import rootTagFilter
import secondTagAdd
import jieba
#import adin
from lstm_data_prepare import data_deal
#To suibin
import os
import new_lda_infer

#for DeepLearning model
from model_server import ModelServer
model_server = ModelServer('10.77.9.21:8145')

p=new_lda_infer.predict_lda()
def tagreflak():
	tagref={}
	for line in open("tag_lev1_lev2"):
		lines=line.strip().split(",")
		tagref[lines[3]]=lines[4]
	return tagref
tagref=tagreflak()

def getdict():
	totaldict={}
	for eachfile in os.listdir("tag_topic_relation"):
		for line in open("tag_topic_relation/{0}".format(eachfile)):
			lines=line.strip().split("\t")
			totaldict[lines[0]]=lines[1]
	return totaldict
totaldict=getdict()
#print totaldict
#To suibin

#param: 1:model 2:word.list 3:corpus 4:predict.result
def loadlev123Data(class_lev123_dict_file):
	class_lev23_dict = {}
	for line in file(class_lev123_dict_file):
		vec = line.strip().split('\t')
		if len(vec) < 5: continue
		if len(vec[-2]) < 2 or len(vec[2]) < 2: continue
		class_lev23_dict.setdefault(vec[-2], [])
		class_lev23_dict[vec[-2]].append(vec[2])
	return class_lev23_dict

class SysInit:
	def __init__(self, model_file, word_dict_file, feature_file, type_root_file, objid_root_file, tagname_id_file,				model_adv_nb_file, model_adv_lr_file, word_dict_adv_file, online_class_file, root_object_file, type2root_file, id2root_file, word2root_file, user_c0123_dict_file, lev1_online_file, seg_word_file, dpl_class_list, gbdt_lr_model_file, gbdt_model_file, onehot_model_file):
		jieba.load_userdict(seg_word_file)
		self.models = joblib.load(model_file)
		self.models_nb_adv = joblib.load(model_adv_nb_file)
		self.models_lr_adv = joblib.load(model_adv_lr_file)
		self.word_dicts = {}
		self.word_adv_dicts = {}
		self.feat_dicts = {}
		self.type_dict = {}
		self.id_dict = {}
		self.root_dict = {}
		self.root_id2name_dict = {}
		self.lev12_dict = {}#1-2id list
		self.class_dict = {}
		self.deep_class_dict = {}
		self.dict_clswrdid = {}
		self.class_lev23id_dict = {}
		self.class_lev23type_dict = {}
		self.class_lev23word_dict = {}
		self.user_c0123_dict = {}
		self.lev1_online_dict = {}
		self.gbdt_lr_model = ''
		self.gbdt_model = ''
		self.onehot_model = ''
		self.gbdt_lr_model = joblib.load(gbdt_lr_model_file)
		self.gbdt_model = joblib.load(gbdt_model_file)
		self.onehot_model = joblib.load(onehot_model_file)
		self._load_word_dicts(word_dict_file)
		self._load_word_adv_dicts(word_dict_adv_file)
		self._load_feat_dicts(feature_file)
		self._load_type_dict(type_root_file)
		self._load_id_dict(objid_root_file)
		self._load_root_dict(tagname_id_file)
		self._load_class_dict(online_class_file)
		self._load_dpl_class_dict(dpl_class_list)
		self._load_root_object(root_object_file)
		self.class_lev23id_dict = loadlev123Data(id2root_file)
		self.class_lev23type_dict = loadlev123Data(type2root_file)
		self.class_lev23word_dict = loadlev123Data(word2root_file)
		self._load_user_c0123_dict(user_c0123_dict_file)
		self._load_lev1_online_dict(lev1_online_file)
		#self.pnd = adin.pnd()

	#windex	word
	def _load_word_dicts(self, word_dict_file):
		for line in file(word_dict_file):
			vec = line.decode('utf-8').strip().lower().split('\t')
			self.word_dicts[vec[1]] = int(vec[0])

	#windex	word
	def _load_word_adv_dicts(self, word_dict_adv_file):
		for line in file(word_dict_adv_file):
			vec = line.decode('utf-8').strip().lower().split('\t')
			self.word_adv_dicts[vec[1]] = int(vec[0])

	#1042015:tagCategory_1004	feature
	def _load_feat_dicts(self, feature_file):
		for line in file(feature_file):
			vec = line.strip().split('\t')
			self.feat_dicts.setdefault(vec[0], [])
			self.feat_dicts[vec[0]].append(vec[1])

	#1042015:tagCategory_1004	type
	def _load_type_dict(self, type_root_file):
		for line in file(type_root_file):
			vec = line.strip().split('\t')
			self.type_dict.setdefault(vec[0], [])
			for i in range(1, len(vec)):
				self.type_dict[vec[0]].append(vec[i])

	#1042015:id	classid
	def _load_id_dict(self, objid_root_file):
		for line in file(objid_root_file):
			vec = line.strip().lower().split('\t')
			if len(vec) != 3: continue
			self.id_dict.setdefault(vec[2], [])
			if vec[0] not in self.id_dict[vec[2]]:
				self.id_dict[vec[2]].append(vec[0])

	#tagCategoryName	1042015:tagCategory_1004
	def _load_root_dict(self, tagname_id_file):
		for line in file(tagname_id_file):
			vec = line.strip().split('\t')
			#if len(vec) <5 or len(vec[0]) < 3 or len(vec[1]) < 3: continue
			if vec[0] not in self.root_dict:
				self.root_dict[vec[0].strip()] = vec[1].strip()#lev1name2id
				#print 'key:%s, value:%s' %(vec[0], vec[1])

			if vec[1] not in self.root_id2name_dict:
				self.root_id2name_dict[vec[1]] = vec[0]#lev1id2name

			self.lev12_dict.setdefault(vec[0], [])
			if vec[4] not in self.lev12_dict[vec[0]]:
				self.lev12_dict[vec[0]].append(vec[4])

	#classlist
	def _load_class_dict(self, online_class_file):
		for line in file(online_class_file):
			vec = line.strip().split('\t')
			if len(vec) != 3: continue
			self.class_dict[vec[0]] = vec[2]
	#classlist
	def _load_dpl_class_dict(self, dpl_class_list):
		for line in file(dpl_class_list):
			vec = line.strip().split('\t')
			if len(vec) != 2: continue
			self.deep_class_dict[vec[0]] = int(vec[1])

	#class object lev3 list
	def _load_root_object(self, root_object_file):
		for line in file(root_object_file):
			vec = line.strip().split('\t')
			self.dict_clswrdid.setdefault(vec[0], [])
			wordid = '@'.join([vec[2], vec[1]])
			self.dict_clswrdid[vec[0]].append(wordid)

	#user c0-c3 uid list
	def _load_user_c0123_dict(self, user_c0123_dict_file):
		for line in file(user_c0123_dict_file):
			vec = line.strip().split('\t')
			if len(vec) != 2: continue
			if vec[0] not in self.user_c0123_dict:
				self.user_c0123_dict[vec[0]] = vec[1]

	#lev1 online dict
	def _load_lev1_online_dict(self, lev1_online_file):
		for line in file(lev1_online_file):
			vec = line.strip().split('\t')
			if len(vec) != 2: continue
			if vec[0] not in self.lev1_online_dict:
				self.lev1_online_dict[vec[0]] = vec[1]

	#return models, word_dicts, feat_dicts, type_dict, id_dict, root_dict, models_nb_adv, models_lr_adv, word_adv_dicts, class_dict


def del_str(str, delstr1, delstr2):
	if str.count(delstr1) < 1:
		return str

	str1 = str.split(delstr1)
	ostr = str1[0]
	for i in range(1, len(str1)):
		str2 = str1[i].split(delstr2)
		if len(str2) >= 2:
			ostr += str2[1] + ' '
		else:
			ostr += str2[0] + ' '
	return ostr.strip()

def del_str2(str, delstr):
	if str.count(delstr) < 1:
		return str

	str1 = str.split(delstr)
	ostr = str1[0]
	for i in range(1, len(str1)):
		ostr += str1[i][7:]

	return ostr.strip()

def preprocess1(str):
	str = del_str(str, '[', ']')
	str = del_str(str, '(', ')')
	str = del_str(str, '（', '）')
	str = del_str(str, '@', ':')
	str = del_str(str, '@', ' ')
	str = str.replace('抱歉，此微博已被删除。查看帮助', '')
	str = str.replace('转发微博', '')
	return str

def procontent(content):
	line = Converter('zh-hans').convert(content.decode('utf-8'))
	line = line.encode('utf-8').lower()
	return preprocess1(line)

def loadData(mid, content, taglist, pic_video):
	x_predict = []
	y_predict = []
	tags_dict = {}
	lev1_tag_dict = {}
	#if len(vec) != 2: continue
	content = procontent(content)
	corpus = ' '.join(list(jieba.cut(content)))
	x_predict.append(corpus)
	#x_predict.append(content)
	y_predict.append(mid)

	tags_list = []
	lev1_list = []
	vec = taglist.strip().split('|')
	for i in range(len(vec)):
		vecs = vec[i].split('@')
		if len(vecs) != 2: continue
		if vecs[0] not in tags_dict and '1042015:' in vecs[0] and 'tagCategory_1004' not in vecs[0]:
			if 'tagCategory_' in vecs[0] and vecs[0] not in lev1_tag_dict:
				lev1_tag_dict[vecs[0]] = float(vecs[1])
				#tags_list.append(vec[i])
				lev1_list.append(vec[i])
			elif 'tagCategory_' not in vecs[0]:
				tags_dict[vecs[0]] = float(vecs[1])
				tags_list.append(vec[i])

	pic_video_list = [int(x) for x in pic_video.split('|')]

	return x_predict, y_predict, tags_dict, content, tags_list, lev1_tag_dict, pic_video_list, lev1_list, corpus

def preprocess(word_dicts, x_predict):
	vectorizer = CountVectorizer(vocabulary=word_dicts)#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
	x_count_vec = vectorizer.fit_transform(x_predict)
	transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
	x_tfidf = transformer.fit_transform(x_count_vec)#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
	x_tfidf = preprocessing.scale(x_tfidf, with_mean=False)
	return x_tfidf

def getRootFeatureNum(rootid, content, feat_dicts):
	num = 0
	if rootid in feat_dicts:
		featlist = feat_dicts[rootid]
		for feat in featlist:
			if feat in content:
				num += 1
				#break
	return num

def getRootTagsNum(rootid, tags, type_dict, id_dict):
	num = 0
	if rootid in type_dict:
		typelist = type_dict[rootid]
		for tag in tags.keys():
			types = tag.split(':')[1].split('_')[0]
			if types in typelist:
				num += 1
				break

	if rootid in id_dict:
		idlist = id_dict[rootid]
		for tag in idlist:
			num += 1
			break

	return num

def	decisionRule(predicty, tags, content, feat_dicts, type_dict, id_dict, root_dict, deep_lev1):
	ret = {}
	otherid = '其他'
	#if predicty not in root_dict: return retid
	rootid = predicty
	weight = 0.6
	#if predicty in root_dict:
	#	rootid = root_dict[predicty]
	#else:
	#	return rootid

	#get root features of rootid(containing tags)
	feat_num = getRootFeatureNum(rootid, content, feat_dicts)
	tags_num = getRootTagsNum(rootid, tags, type_dict, id_dict)

	#if ((predicty == '情感两性' or predicty == '搞笑幽默') or deep_lev1) and feat_num+tags_num < 1:
	if (predicty == '情感两性' or predicty == '搞笑幽默') and feat_num+tags_num < 1:
		feat_num = 1

	#feat_tags_num = getFeaturesOfRoot(rootid, content, feat_dicts, tags, type_dict, id_dict)
	#if else rule decision begin
	#1\food, health(health feature to decide) >= 2
	#2\health, hospital(disease to decide) >= 1
	#if predicty == '健康医疗' and tags_num < 1:
	#	return otherid
	#3\beauty, fashion(beauty feature to decide) >= 2
	#4\food, travel(##,[],object pos to decide) >= 1
	#5\travel, photo(photo feature to decide) >= 2
	#6\mv, tv, show(type/object to decide)
	#7\music(music feature to decide)
	#8\book, feel(book feature to decide)
	#9\cartoon, game(object/type/feature to decide)
	#10\mv/tv/show/music/model+person->entertainmentStar
	#11\military, social(military feature to decide)
	#0\
	if feat_num + tags_num < 1:
		#get all domain features
		return ret
		#return otherid, weight
	#decision end
	weight += 0.1*(feat_num + tags_num)
	if weight > 1.0:
		weight = 0.95
	if rootid not in ret:
		ret[rootid] = weight

	#return rootid, weight
	return ret

def predictlev3(class_name_list, dict_clswrdid, content):
	ret = {}
	for class_name, class_weight in class_name_list.items():
		if class_name not in dict_clswrdid: return ret
		taglist = dict_clswrdid[class_name]
		for wordid in taglist:
			#print wordid
			word, id = wordid.split('@')
			if word in content:
				cnt = content.count(word)
				pos = content.find(word)
				weight = class_weight + cnt*1.0*(1-1.0*pos/len(content))
				if weight > 1.0:
					weight = 1.0
				if weight > 0.5 and id not in ret:
					ret[id] = weight
					#ret.append('@'.join([id, '%f'%weight]))
	return ret

def mergeLev3Id(taglist, tagdict, lev3_tag_dict):
	for key, val in lev3_tag_dict.items():
		if key not in tagdict:
			tagdict[key] = val
			taglist.append('@'.join([key, '%f' %val]))
	return taglist, tagdict

def is_filter(taglist, lev1_dicts):
	for tags in	taglist:
		tagsid = tags.split('@')[0]
		if 'tagCategory' in tagsid and tagsid in lev1_dicts:
			#print 'tags:%s' %tags
			return False
	return True

def mergeLev1Id(taglist, lev1_tag_dict, lev1_name2id_dict):
	has_humor = False
	for key, val in lev1_tag_dict.items():
		val = float(val)
		key = key.strip().split('\t')[0]
		key2 = key
		#ls = {}
		if 'tagCategory' not in key:
			try:
				if key in lev1_name2id_dict:
					key2 = lev1_name2id_dict[key]
				#for tname, tid in lev1_name2id_dict.items():
				#	print 'class name:%s, predict name:%s' %(tname, key)
				#	if tname == key:
				#		key2 = tid
				#		break
			except Exception as e:
				print 'key is :%s,key2 is :%s' %(key, key2)
		#print 'target tagCategory_id:%s, name:%s' %(key2, key)
		if 'tagCategory_020' in key2:
			has_humor = True
		ostr = '@'.join([key2, '%f' %val])
		taglist.append(ostr)
	return taglist, has_humor

def get_tag_id_type(taglist):
	tag_id_list = []
	tag_type_list = []

	#tag_id_list = [x.split('@')[0] for x in taglist.strip().split('|')]
	tag_id_list = [x for x in taglist.keys()]
	tag_type_list = list(set([tagid.split(':')[1].split('_')[0] for tagid in tag_id_list]))

	return tag_id_list, tag_type_list

#lev2 recognize pattern
def get_type_num(lev2name, object, tag_type_list):
	type_cnt = 0

	for types in tag_type_list:
		if types in object.class_lev23type_dict[lev2name]:
			type_cnt += 1
	return type_cnt

def get_id_num(lev2name, object, tag_id_list):
	id_cnt = 0

	for id in tag_id_list:
		if id in object.class_lev23id_dict[lev2name]:
			id_cnt += 1
	return id_cnt

def get_word_num(lev2name, object, content):
	feat_cnt = 0

	for feat in object.class_lev23word_dict[lev2name]:
		feat_cnt += content.count(feat)
	return feat_cnt


def predictlev2(taglist, oridict, paramInit, class_name_list, tagdict, content):#利用数据结构，加载原始1-2,2-3对照表;根据type2root, id2root, word2root的2级到3级/word文件；
	lev2ret = {}
	tag_id_list, tag_type_list = get_tag_id_type(tagdict)
	for class_name, class_weight in class_name_list.items():
		lev2tmp = {}
		if class_name not in paramInit.lev12_dict: continue
		for lev2name in paramInit.lev12_dict[class_name]:
			feat_cnt = 0
			type_cnt = 0
			objid_cnt = 0

			#type to recognize lev2
			if lev2name in paramInit.class_lev23type_dict:
				type_cnt = get_type_num(lev2name, paramInit, tag_type_list)

			#id to recognize lev2
			#if type_cnt <= 0:
			if type_cnt <= 0 and lev2name in paramInit.class_lev23id_dict:
				objid_cnt = get_id_num(lev2name, paramInit, tag_id_list)

			#word to recognize lev2
			if feat_cnt <= 0 and lev2name in paramInit.class_lev23word_dict:
				feat_cnt = get_word_num(lev2name, paramInit, content)

			if feat_cnt + type_cnt + objid_cnt <= 0: continue
			weight = 0.4+ 0.5*class_weight + 0.8*(0.2*feat_cnt+0.6*type_cnt+0.2*objid_cnt)
			if weight > 1.0:
				weight = 1.0
			if weight > 0.5 and lev2name not in lev2ret and lev2name not in oridict:
				lev2tmp[lev2name] = weight

		if len(lev2tmp) >= 1:
			lev2tmplist = sorted(lev2tmp.iteritems(), key=lambda d:d[1], reverse = True)
			cnt = 0
			for (lev2name, weight) in lev2tmplist:
				cnt += 1
				if cnt > 2: break
				lev2ret[lev2name] = weight
				lev2idw = '@'.join([lev2name, '%f' %weight])
				taglist.append(lev2idw)
	return taglist, lev2ret

def transform_lev1_id2name(lev1_tag_dict, root_id2name_dict):
	lev1_dict = {}
	for key, val in lev1_tag_dict.items():
		if 'tagCategory_' in key and key in root_id2name_dict:
			key = root_id2name_dict[key]
		lev1_dict[key] = val
	return lev1_dict

def test_output(odict, flag):
	print flag
	for key, val in odict.items():
		ret = '\t'.join([key, '%f'%val])
		print ret

def getStarCnt(lev1_dict, lev23_list, content):
	cnt = 0
	cnt_model = 0
	for idw in lev23_list:
		idwt = idw.split('@')
		if len(idwt) !=2: continue
		if ('moviePerson' in idw or 'musicPerson' in idw) and float(idwt[1]) >= 0.5:
			#print 'add star person:', idw
			cnt += 1
		if 'modelPerson' in idw and float(idwt[1]) >= 0.5:
			cnt_model += 1
	return cnt, cnt_model

def addDelStarLev1(lev1_dict, lev23_list, content, starlev1_flag):
	#test_output(lev1_dict, "before:\n")
	cnt, cnt_model = getStarCnt(lev1_dict, lev23_list, content)

	if cnt >= 1 and not starlev1_flag:
		weight = 0.5 + cnt * 0.1
		if weight > 1.0:
			weight = 1.0
		#print 'star lev1 add'
		lev1_dict['1042015:tagCategory_050'] = weight

	elif cnt + cnt_model <= 0 and starlev1_flag:
		#print 'star lev1 del'
		lev1_dict.pop('1042015:tagCategory_050')
	#test_output(lev1_dict, "end\n")

	return lev1_dict

def is_star(starls, lev1dict):
	for star in starls:
		if star in lev1dict:
			return True
	return False

def addDelLev1(lev1_dict, lev23_list, content, pic_video_list):
	starlist = ['音乐', '电影', '电视剧', '综艺节目', '时尚']
	add_star = is_star(starlist, lev1_dict)

	if '1042015:tagCategory_050' in lev1_dict:
		lev1_dict = addDelStarLev1(lev1_dict, lev23_list, content, 1)
	elif add_star:
		lev1_dict = addDelStarLev1(lev1_dict, lev23_list, content, 0)
	return lev1_dict

#def decision(line, models, word_dicts, feat_dicts, type_dict, id_dict, root_dict, models_nb_adv, models_lr_adv, word_adv_dicts, online_class_list):
def decision(line, paramInit, uidability, is_comment):
	begin = time.time()
	has_lev2 = False
	predict_lev1 = False
	uid_abi_lev1 = False
	deep_lev1 = False
	vec = line.strip().split('\t')
	ostr = list(vec[3].split('|'))
	ostr_ret = ['1042015:tagCategory_1004@0.5']
	if len(vec) != 5:
		return ostr_ret, False, False, False, False, deep_lev1

	has_lev1 = False
	x_predict, y_predict, tagdict, content, taglist, lev1_tag_dict, pic_video_list, lev1_list, segwords = loadData(vec[0], vec[2], vec[3], vec[4])
	content2 = content

	#add lev1 for star//对于打错的一级，进行策略纠正；针对没有打上的，进行策略补打//TODO//尤其是娱乐明星
	#lev1_tag_dict = addDelLev1(lev1_tag_dict, taglist, content, pic_video_list)
	#if 'tagCategory_' not in vec[2] or 'tagCategory_1004' in vec[2]:
	if len(lev1_tag_dict) > 0:#直接推送线上打标签结果//TODO
		#//TODO 改进修正策略
		#add by suibin.add topic tag
		result=p.entra(segwords,taglist)
		if len(result):
			for each in result:
				if each in totaldict:
					if totaldict[each] in tagref:
						tag2=tagref[totaldict[each]]
						tagw=float(result[each])+0.5
						if tagw > 1.0:
							tagw = 1.0
						sline=tag2+"@"+str(tagw)
						print 'lev2 is:', sline
						taglist.append(sline)
						has_lev2 = True
		return taglist+lev1_list, has_lev2, False, False, False, deep_lev1

	if len(lev1_tag_dict) <= 0:
		has_lev1 = True
		#advertisement predict
		x_tfidf_adv = preprocess(paramInit.word_adv_dicts, x_predict)
		grd_enc_data_predict = paramInit.onehot_model.transform(paramInit.gbdt_model.apply(x_tfidf_adv)[:, :, 0])
		#print x_tfidf_adv.shape, grd_enc_data_predict.shape
		x_predict_new = scipy.sparse.hstack((x_tfidf_adv, grd_enc_data_predict)).A
		#x_predict_new = bmat([[x_tfidf_adv, grd_enc_data_predict]]).toarray()
		#print '---ok---', x_tfidf_adv.shape, grd_enc_data_predict.shape, x_predict_new.shape
		y_gbdt_lr_adv = paramInit.gbdt_lr_model.predict(x_predict_new)
		if y_gbdt_lr_adv[0] == '广告' or (('券' in content or '卷后' in content or '领卷' in content or '优惠卷' in content or '卷.后' in content or '￥' in content) and '证券' not in content) or '代购' in content or '现货' in content or '直邮' in content or '持家省钱联盟' in content or '原价' in content or '劵' in content:
		#y_nb_adv = paramInit.models_nb_adv.predict(x_tfidf_adv)
		#y_lr_adv = paramInit.models_lr_adv.predict(x_tfidf_adv)
		#if y_nb_adv[0] == y_lr_adv[0] and y_nb_adv[0] == '广告':
			y_gbdt_lr_adv[0] = '广告'
			print '--------------predict target is advertisement:%s' %line
			return ostr_ret, False, False, False, True, deep_lev1

		#non adv predict
		x_tfidf = preprocess(paramInit.word_dicts, x_predict)
		y = paramInit.models.predict(x_tfidf)
		iterval = time.time() - begin
		print 'classify get tag time:%d, mid:%s' %(iterval, vec[0])

		####add LSTM model for short text
		if y[0] == '其他':
			content = data_deal(content)
			if '身边事#' in content:
				pos = content.find('身边事#')
				if pos <= 13:
					content = '#'.join(content.split('#')[2:])
			content = content.strip().replace(' ','').replace('\t', '').replace('#', '').replace('?', '？')
			#print 'preprocess before:', content
			content = del_str2(content, 'http://t.cn/')
			#print 'preprocess after:', content
			if ('t.cn' in content and len(content) > 52) or ('t.cn' not in content and len(content) > 33):
				try:
					y[0] = model_server.getTag(content).split(':')[1]
				except Exception,e:
					pass
				if y[0] != '其他' and y[0] in paramInit.deep_class_dict and paramInit.deep_class_dict[y[0]] != 1:
					y[0] = '其他'
			if y[0] != '其他' or 'Bad request' not in y[0]:
				deep_lev1 = True
				print 'deeplearning model predict class:%s' %y[0]
		####end
		
		###add trick tags20171017
		if y[0] == '其他':
			pass
		###end

		print 'classify:%s' %y[0]
		uid_ability = False
		uid_ability_id = ''
		uid_ability_wi = 0.0
		if y[0] == '其他':
			#//TODO 补打一级标签
			#vec[0]--mid, vec[1]--uid
			uid_ability_id, uid_ability_wi = uidability.processNd(segwords, vec[1], vec[0], is_comment)
			if uid_ability_id == "" and uid_ability_wi == 0: return ostr_ret, False, False, False, False, deep_lev1
			uid_ability = True
			uid_abi_lev1 = True
			if is_comment:
				print 'uid interest id:%s' %uid_ability_id
			else:
				print 'uid ability id:%s' %uid_ability_id

		if uid_ability:
			lev1_tag_dict[uid_ability_id] = uid_ability_wi
		else:
			predict_lev1 = True
			lev1_tag_dict = decisionRule(y[0], tagdict, content2, paramInit.feat_dicts, paramInit.type_dict, paramInit.id_dict, paramInit.root_dict, deep_lev1)
			lev1_tag_dict = addDelLev1(lev1_tag_dict, taglist, content2, pic_video_list)
			if len(lev1_tag_dict) <= 0:
				deep_lev1 = False
				#print 'ret:%s' %('\t'.join(ostr))
				return ostr_ret, False, False, False, False, deep_lev1
			iterval = time.time() - begin
			print 'filter get tag time:%d, mid:%s' %(iterval, vec[0])

	if len(lev1_tag_dict) > 0:
		lev1_tag_dict = transform_lev1_id2name(lev1_tag_dict, paramInit.root_id2name_dict)

		pos = content2.find('$#$&$#$&')
		if pos > 0:
			content2 = content2[:pos]

		lev3_tag_dict = predictlev3(lev1_tag_dict, paramInit.dict_clswrdid, content2)
		taglist, lev3_tag_dict = mergeLev3Id(taglist, tagdict, lev3_tag_dict)

		taglist, lev2_tag_dict = predictlev2(taglist, tagdict, paramInit, lev1_tag_dict, lev3_tag_dict, content2)
		#print 'lev11 is :%s' %('|'.join(list(lev1_tag_dict.keys())))
		taglist, has_humor = mergeLev1Id(taglist, lev1_tag_dict, paramInit.root_dict)
		#print 'new taglist:%s' %('\t'.join(taglist))

		if has_lev1:
			taglist = rootTagFilter.humor(taglist, pic_video_list[2], pic_video_list[3])
		elif has_humor:
			taglist = secondTagAdd.humor(taglist, pic_video_list[2], pic_video_list[3])
		iterval = time.time() - begin
		#print 'lev23 get tag time:%d' %iterval
		print 'lev23 get tag time:%d, mid:%s' %(iterval, vec[0])
		#TODO suibin
		#input:taglist, segwords
		#print "shikunsegwords  "+segwords
		#print taglist
		result=p.entra(segwords,taglist)
		if len(result):
			for each in result:
				#print each
				if each in totaldict:

					if totaldict[each] in tagref:

						tag2=tagref[totaldict[each]]
						tagw=float(result[each])+0.5
						if tagw > 1.0:
							tagw = 1.0
						sline=tag2+"@"+str(tagw)
						#sline=tag2+"@"+str(float(result[each])+0.5)
						print 'lev2 is:', sline
						taglist.append(sline)
						has_lev2 = True
		return taglist, has_lev2, predict_lev1, uid_abi_lev1, False, deep_lev1
	return ostr_ret, False, False, False, False, deep_lev1
