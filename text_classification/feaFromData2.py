#!/usr/bin/env python
#coding=utf-8

import sys, os, shutil
import re, urllib, urllib2, json,redis
import time,socket
from lib import get_from_api
from multiprocessing import Queue,Process,Pool
import mylogging
import Queue
import threading
from write_token import *

flist = {}
queue = Queue.Queue(maxsize = int(sys.argv[1]))
log=mylogging.getLogger('fea')

#获取图片信息
def getPicSize(pid):
	''' 
	Compute the width and height of a picture by its picture id
	>>> getPicSize('81f892c2jw1eq45l3wq3bj20u01hck12')
	(1080, 1920)
	'''
	assert isinstance(pid,str) or isinstance(pid,unicode)
	str36='0123456789abcdefghijklmnopqrstuvwxyz'
	charMap = dict((s,i) for i,s in enumerate(str36))
	if len(pid) < 32:
		return (0,0)
	w = charMap[pid[23]]*36*36 + charMap[pid[24]]*36 + charMap[pid[25]]
	h = charMap[pid[26]]*36*36 + charMap[pid[27]]*36 + charMap[pid[28]]
	return (w, h)

#获取图片
def get_gif_longPic( pic_ids ):
	gif_num = 0
	long_pic_num = 0
	for pid in pic_ids:
		#是否是gif图
		if pid[21] == 'g':
			gif_num += 1
		#是否是大图
		p_width, p_height = getPicSize( pid )
		if p_width>0 and p_height>0 and p_height >= 3*p_width:
			long_pic_num += 1
	return (gif_num,long_pic_num)


#批量接口获取信息
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
			log.warn('feaFromData-feaFromMblogInfo:获取微博信息,接口失败,e=%s,mids_list=%s'%(e,'|'.join(mids_list)))
			continue

	#获取微博信息
	for status in j_v.get('statuses',[]):
		tmp_data = {}
		try:
			tmp_data['mid'] = status['mid'].encode('utf8')
			text = status['text'].encode('utf-8').replace('\n',' ').replace('\t',' ')
			if 'longText' in status.keys():
				ltext=status['longText'].get('longTextContent','').encode('utf-8').replace('\n',' ').replace('\t',' ')
				if len(ltext)>len(text):
					text=ltext
			tmp_data['text'] = text
			if '抱歉，此微博已被删除' in text:
				continue
			tmp_data['uid'] = status['user']['idstr'].encode('utf8')
			tmp_data['gif_num'], tmp_data['long_pic_num'] = get_gif_longPic( status['pic_ids'] )
			tmp_data['pic_num'] = len( status['pic_ids'] )
			data.append( tmp_data )
		except Exception as e:
			log.warn('feaFromData-feaFromMblogInfo:微博信息解析json失败,e=%s'%e)
			continue
	return data

#判断oid是否再tag中 
def notInTags( oid, tags ):
	for idwei in tags:
		id,wei = idwei.split('@')[:2]
		if id == oid:
			return False
	return True

#处理微博文本，获取表情、昵称、短链等信息
def textPro( text):
	text_tmp=text.decode('utf-8')
	re_short_url = re.compile(r'http://t\.cn/\w{,7}')

	#短链
	urls = re_short_url.findall( text )
	video_num_inner=0
	video_num_outer=0
	for url in urls:
		text_tmp=text_tmp.replace(url.decode('utf-8'),'')
		if 'zOXAaic' in url:
			continue
		print 'get short url:%s' %url
		url_long,object_type,object_id = get_from_api.aysShortUrl( url )
		print 'get short url ok:%s' %url
		if object_type == 'video':
			if 'miaopai' in url_long:
				video_num_inner += 1
			else:
				video_num_outer += 1
	return video_num_inner,video_num_outer

#批量获取
def getMblogFea( mids_list, mid_info_dict ):
	mblogInfoResList = feaFromMblogInfo( mids_list )
	for mblogInfoRes in mblogInfoResList:
		this_mid = mblogInfoRes[ 'mid' ]
		if this_mid not in mid_info_dict:
			log.info('feaFromData-getMblogFea:%s not in dict '%this_mid)
			continue
		#mid,uid,text,pic,gif,longpic,ivedio,ovedio,tags,hash
		mid_info_dict[ this_mid ][1]=mblogInfoRes['uid']
		mid_info_dict[ this_mid ][2]=mblogInfoRes['text']
		mid_info_dict[ this_mid ][3]=mblogInfoRes['pic_num']
		mid_info_dict[ this_mid ][4]=mblogInfoRes['gif_num']
		mid_info_dict[ this_mid ][5]=mblogInfoRes['long_pic_num']
		mid_info_dict[ this_mid ][6],mid_info_dict[ this_mid ][7]=textPro(mblogInfoRes['text'])
	
#处理文件
def process_file( input_file, output_file ):
	fOutput = open( output_file, 'w')
	mid_list = []
	mid_info_dict = {}
	print 'begin process file:%s' %input_file
	for line in open( input_file ):
		item = line.strip().split( '\t' )
		try:
			if item[0] in mid_info_dict:
				continue
			mid = item[0]
			to_ids = item[1]
			simhash = item[2]
			mid_info_dict[ mid ] = [mid,'','',0,0,0,0,0,to_ids,simhash]
			#mid,uid,text,pic,gif,longpic,ivedio,ovedio,tags,hash

			#批量调取
			mid_list.append( mid )
			if len( mid_list ) >= 50:
				getMblogFea( mid_list, mid_info_dict )
				mid_list = []
		except Exception as e:
			log.warn('feaFromData-process_file:e=%s'%e)
	#处理剩余数据
	if len( mid_list ) > 0:
		getMblogFea( mid_list, mid_info_dict )
	#写文件
	for mid in mid_info_dict:
		info = mid_info_dict[ mid ]
		mid=info[0]
		text=info[2]
		to_ids=info[-2].split('|')
		
		#if text == '' or len(to_ids)<1:
		if text == '':
			log.info(mid+' text or to_ids is null')
			continue
		
		if len(to_ids) < 1:
			info[-2] = '1042015:tagCategory_1004@0.450'

		try:
			fOutput.write( '\t'.join(map(str,info)) + '\n' )
			fOutput.flush()
		except Exception as e:
			log.warn('feaFromData-process_file:写文件失败,e=%s,mid=%s'%(e, mid))

	fOutput.close()
	print 'process file success:%s' %input_file

def consumer(indir, outdir, bakdir):
	while True:
		f = queue.get()
		print 'get file from queue:%s' %f
		try:
			input_file = indir + f 
			output_file = outdir + '_' + f
			ifs = input_file
			if not os.path.exists(ifs):continue
			process_file( input_file, output_file )
			os.rename( outdir+'_'+f, outdir+f )
			shutil.move( indir + f, bakdir + f )

		except Exception as e:
			log.warn('feaFromData-process_thread:input[%s] outdir[%s] bakDir[%s] is wrong:e[%s] ' % (\
										indir, outdir, bakdir, e))

if __name__=='__main__':
	indir = '../common_data/content_tag/'
	outdir = '../data_fea/content_tag/'
	bakdir = '../data_back/common_data/content_tag/'

	#paths=[ 'content_tag','top_user']
	thread_consumer1 = [threading.Thread(target = consumer, args = (indir, outdir, bakdir)) for i in range(int(sys.argv[2]))]
	for ithread in thread_consumer1:
		ithread.start()
	
	while True:
		try:
			filelist = [ f for f in os.listdir(indir) if not f.startswith('.') and not f.startswith('_') ]
			for f in filelist:
				input_file = indir + f 
				output_file = outdir + '_' + f
				###当前时间-最后修改时间>10s

				ifs = input_file
				if not os.path.exists(ifs):continue
				statinfo = os.stat(ifs)
				delta = int(time.time()) - int(statinfo.st_mtime)
				if delta < 20: continue

				if len(flist) >= 200000: flist.clear()
				if f not in flist:
					queue.put(f)
					print 'put queue file:%s' %f
					flist[f] = 1

		except Exception as e:
			print 'exception:',e

