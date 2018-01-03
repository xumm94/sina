#!/usr/bin/env python
#coding=utf-8
import re, urllib, urllib2, json,redis
from write_token import *
import requests
import cPickle as pickle


def get_pic_class( pid, tags_dict):
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
			if term['weight']>=0.75 and "1042015:tagCategory_1004" != term["object_id"] and term["object_id"] in tags_dict:
				ids.append(str(tags_dict[term["object_id"]]) + '@' + str(term['weight']))
	except Exception as e:
		pass
		#print('get_pic_class pid is %s, error = %s' % (pid,traceback.format_exc()))
	return ids


def feaFromMblogInfo( mids_list ):
	data = []
	j_v = {}
	for i in range(3):
		try:
			url = "http://i2.api.weibo.com/2/statuses/show_batch.json?source=1586188222&simplify=1&isGetLongText=1&ids=%s"%(','.join( mids_list ))
			header = {"Authorization":get_header()}
			print 'get pic of midlist begin:'
			req = urllib2.Request(url,None,header)
#			req = "http://10.77.136.59:9001/statuses/show_batch.json?simplify=1&isGetLongText=1&ids=" + ','.join( mids_list )
			htmlret = urllib2.urlopen(req, timeout=159)
			print 'get pic of midlist success'
			html = htmlret.read()
			htmlret.read(1)
			htmlret.close()
			j_v = json.loads(html)
			if 'statuses' not in j_v:
				continue
			else:
				break
		except Exception as e:
			#log.warn('feaFromData-feaFromMblogInfo:获取微博信息,接口失败,e=%s,mids_list=%s'%(e,'|'.join(mids_list)))
			continue

	#获取微博信息
	for status in j_v.get('statuses',[]):
		tmp_data = {}
		try:

			#tmp_data['mid'] = status['mid'].encode('utf8')
			text = status['text'].encode('utf-8').replace('\n',' ').replace('\t',' ')
			if 'longText' in status.keys():
				ltext=status['longText'].get('longTextContent','').encode('utf-8').replace('\n',' ').replace('\t',' ')
				if len(ltext)>len(text):
					text=ltext
			tmp_data['text'] = text
			if '抱歉，此微博已被删除' in text:
				continue
			#tmp_data['uid'] = status['user']['idstr'].encode('utf8')
			#tmp_data['gif_num'], tmp_data['long_pic_num'] = get_gif_longPic( status['pic_ids'] )
			#tmp_data['pic_num'] = len( status['pic_ids'] )
			tmp_data['pic'] = [str(pic) for pic in status['pic_ids']]
			data.append( tmp_data )
		except Exception as e:
			#log.warn('feaFromData-feaFromMblogInfo:微博信息解析json失败,e=%s'%e)
			continue
	return data


if __name__ =='__main__':

	input_file = open('data_part', 'r')
	tags_dict = pickle.load(open('tags_dict.pkl', 'rb'))

	with open('data_output', 'w') as f:
		for line in input_file:
			columns = line.split('\t')
			mid = [columns[0]]
			data = feaFromMblogInfo(mid)
			if(len(data) != 1):
				continue

			pids = data[0]['pic']
			pic_tags = []

			for pid in pids:
				pic_tags.extend(get_pic_class(pid, tags_dict))

			if len(pic_tags) == 0:
				content = line.strip('\r\n') + '\t' + 'no_pic_tag'
			else:
				content = line.strip('\r\n') + '\t' + '|'.join(pic_tags)
			
			content += '\n'
			f.writelines(content)

	input_file.close()






