#!/usr/bin/env python
#coding=utf-8

import sys, os, traceback,threading
import re, urllib, urllib2, json,redis
import time,socket,requests

class MyThread(threading.Thread):
	def __init__(self,func,args):
		super(MyThread,self).__init__()
		self.func = func
		self.args = args

	def run(self):
		self.result = self.func(self.args)

	def get_result(self):
		try:
			return self.result
		except Exception:
			return None

#多线程获取结果
def func_thread(func,args):
	thrd = []
	results = []
	for key in args:
		t = MyThread(func,args=key)
		thrd.append(t)
		t.start()
	for t in thrd:
		t.join()
		results.append(t.get_result())
	return results

#图片打标签			
def get_pic_class( pid ):
	ids=[]
	try:
		req = "http://10.77.136.59:5003/image_classify/v1.5/query?url=http://ww2.sinaimg.cn/bmiddle/" + pid
		res = requests.get(req, stream=False, timeout=5)
		html = res.text
		if 'error' in html:
			return ids
		res.close
		j_v = json.loads(html)
		result=j_v['results']
		results=[pid]
		for term in result:
			results.append(term["object_id"])
			if term['weight']>=0.75 and "1042015:tagCategory_1004" != term["object_id"]:
				ids.append(term["object_id"])
	except Exception as e:
		print('get_pic_class pid is %s, error = %s' % (pid,traceback.format_exc()))
	return ids

#图片打标签结果解析
def addPicTags(pid,to_ids,uidCate):
	root_id = []
	new_to_id = []
	to_id = []
	bad_ids = ["1042015:tagCategory_028","1042015:tagCategory_031"]
	for term in to_ids:
		to_id.append(term.split('@')[0])
		new_to_id.append(term)
		if "1042015:tagCategory" in term:
			root_id.append(term.split('@')[0])
	#有一级的微博不打标签
	if len(root_id) >1 or len(root_id) == 1 and "1042015:tagCategory_1004" not in root_id:
		return new_to_id
	ids=func_thread(get_pic_class,pid)
	add_ids={}
	for term in ids:
		for id in term:
			if id not in add_ids:
				add_ids[id] = 0
			add_ids[id] += 1
	ucat = [term[0] for term in uidCate]
	for term in add_ids:
		if add_ids[term] < 4:
			continue
		if term == "1042015:tagCategory_009":
			if "1042015:tagCategory_009" not in ucat:
				continue
		elif term == "1042015:tagCategory_039":
			if "1042015:tagCategory_039" not in ucat:
				continue
		elif term == "1042015:tagCategory_025":
			bad_cat = ["1042015:tagCategory_060","1042015:tagCategory_036","1042015:tagCategory_019"]
			if len(set(bad_cat) & set(ucat)) > 0:
				continue
		elif term == "1042015:tagCategory_012":
			bad_cat = ["1042015:tagCategory_032"]
			if len(set(bad_cat) & set(ucat)) > 0:
				continue
		if term not in to_id and term not in bad_ids:
			to_id.append(term)
			new_to_id.append(term+"@0.7777")
	return new_to_id

if __name__=='__main__':
	pic_id='006szs19ly1fmabpncpz7j30zk0k0q93|006szs19ly1fmabpsnnxoj30zk0k0jz9|006szs19ly1fmabpkqkjmj30za0jq43n|006szs19ly1fmabpuc8q6j30zc0jv7bw'
	to_id='1042015:keyWord_64109@0.450881235154@0|1042015:keyWord_154813@0.319289786223@0|1042015:abilityTag_571@0.634774346793@0|1042015:keyWord_154642@0.320714964371@0|1042015:keyWord_154658@0.449456057007@0|1042015:keyWord_154675@0.301@0|1042015:keyWord_57531@0.311451306413@0|1042015:abilityTag_1185@0.6666@1|1042015:park_10014651204poi28480@1.0@0|1042015:keyWord_155533@0.288885985748@0|1042015:keyWord_115868@0.442330166271@0|1042015:tagTopic_403@0.131578947368|1042015:tagTopic_1000@0.128947368421'
	to_ids=to_id.split('|')
	pic_ids=pic_id.split('|')
	uidCate=[['1042015:tagCategory_016',60.0]] #限定用户能力进行打标签
	if len(pic_ids)>3:
		to_ids = addPicTags(pic_ids,to_ids,uidCate)
		print to_ids
