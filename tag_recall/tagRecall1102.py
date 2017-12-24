#!/usr/bin/env python
# encoding: utf-8

#import simplejson as json
from predict_online1102 import *
from put2table import *
#from get_comment3 import *
from deal_blog_comment import *
from uidAbilityTag import *
import threading
import Queue
from adin0320 import *
import os, sys, time, urllib2
from Write2Plat import *
from tags_trick import *
reload(sys)

def is_filter(cols):
	if len(cols) != 5: return True
	for col in cols:
		if len(col) <= 3:
			return True
	return False

def has_lev1(cols):
	taglist = cols[3].split('|')
	lev1list = [tag.split('@')[0] for tag in taglist if 'tagCategory_1004' not in tag]
	for lev1id in lev1list:
		if 'tagCategory_' in lev1id:
			return True

	return False

def getlist(taglist):
	retls = []
	for tag in taglist.split('|'):
		tagvec = tag.split('@')
		if float(tagvec[1]) >=0.5009:
			retls.append(tag)
	return '|'.join(retls)

def tag_filter(taglist, wbstr, pvstr):
	flag = False
	tagstr = getlist(taglist)
	if 'tagCategory_025' in taglist:
		if 'park_' not in taglist and 'travel' not in taglist:
			flag = True
	elif 'tagCategory_041' in taglist:
		if 'food' not in taglist and 'cuisine' not in taglist and 'restaurantCompany' not in taglist and 'wine' not in taglist:
			flag = True
	elif 'tagCategory_019' in taglist:
		if 'stock' not in taglist and 'finace' not in taglist:
			flag = True
	elif 'tagCategory_012' in taglist:
		if '1042015:sport' not in taglist or 'sportBrand_' in taglist:
			flag = True
	elif 'tagCategory_050' in taglist:
		if 'musicPerson_' not in taglist and 'moviePerson_' not in taglist and 'modelPerson_' not in taglist:
			flag = True
	elif 'tagCategory_018' in taglist:
		if 'http://t.cn/' in wbstr:
			flag = True
		elif int(pvstr.split('|')[3]) >= 1:
			flag = True
		elif '二哈' in wbstr or '喵喵' in wbstr or '兔子' in wbstr or '佛' in wbstr or '《' in wbstr or '——' in wbstr:
			flag = True
	elif 'tagCategory_027' in taglist:#星座命理
		if '1042015:edu' in tagstr or '美女' in wbstr or '帅哥' in wbstr or '妹子' in wbstr or '校花' in wbstr or '征婚' in wbstr or '交友' in wbstr or '月老牵线' in wbstr or '女嘉宾' in wbstr:
			flag = True
	elif 'tagCategory_031' in taglist:#数码
		if '1042015:3c' not in taglist:
			flag = True
	elif 'tagCategory_029' in taglist:#汽车//Done
		if '1042015:car' not in taglist:
			flag = True
	#sportsFitnessChild  fitnessConcept slimingMethod  slimingChild
	#瘦身，减脂，肌肉，脂肪，梨形身材，运动，晨跑，拉伸，燃脂
	elif 'tagCategory_045' in taglist:#运动健身
		if 'sportsFitness' not in tagstr and 'fitness' not in tagstr and 'sliming' not in tagstr:
			flag = True
		elif '瘦身' not in wbstr and '减脂' not in wbstr and '肌肉' not in wbstr and '脂肪'  not in wbstr and '身材' not in wbstr and '运动' not in wbstr and '晨跑' not in wbstr and '拉伸' not in wbstr and '燃脂' not in wbstr and '运动' not in wbstr and '锻炼' not in wbstr:
			flag = True
	elif 'tagCategory_028' in taglist:#母婴育儿//DONE
		if 'babyMother' not in tagstr:
			flag = True
	elif 'tagCategory_024' in taglist:#教育
		if '1042015:edu' not in tagstr or ('美女' in wbstr or '帅哥' in wbstr or '妹子' in wbstr or '校花' in wbstr or '征婚' in wbstr or '交友' in wbstr or '月老牵线' in wbstr or '女嘉宾' in wbstr):
			flag = True
	elif 'tagCategory_006' in taglist:#健康医疗
		if 'foodMenu' in tagstr:
			flag = True
	elif 'tagCategory_035' in taglist:#科学科普
		if '1042015:space' not in tagstr and '1042015:astroTele' not in tagstr and '1042015:science' not in tagstr:
			flag = True

	return flag

def lev1_filter(filter_lev1, paramInit, has_lev1_flag, has_lev2):
	if has_lev1_flag or has_lev2: return False

	for lev1idw in filter_lev1:
		lev1id = lev1idw.split('@')[0]
		if lev1id in paramInit.class_dict and paramInit.class_dict[lev1id] == '1':
			#print lev1id, "not filter"
			return False
	#print 'filter taglist:%s' %('|'.join(filter_lev1))
	return True

def processFile(midlist, paramInit, uidInit, uidAbility, ifs, ofs, bfs, tfs = '', tfsx = ''):
	#print 'begin ifs:%s, ofs:%s, bfs:%s, tfs:%s, tfsx:%s' %(ifs, ofs, bfs, tfs, tfsx)
	ofs = open(ofs, 'w')
	bfs = open(bfs, 'w')
	if len(tfs) > 0 and '__1' not in ifs:
		tfsx = open(tfsx, 'w')
	begin = time.time()
	nrow = 0
	okcn = 0
	for line in file(ifs):
		try:
			nrow += 1
			oldline = line
			vec = line.strip().split('\t')
			if len(vec) < 10:
				#print 'col len <10'
				continue #mid,uid,text,图片数，动图数，长图，内生视频，外站视频，标签，哈希值
			#if vec[1] not in paramInit.user_c0123_dict: continue
			#print vec[0]#mid
			#print vec[2]#content
			'''
			print vec[1]#uid
			print vec[24]#taglist
			print vec[7], vec[9], vec[8], vec[10], vec[11]#pic+video
			print vec[26]#simhash
			'''
			#simhash = vec[26]
			#videonum = int(vec[10]) + int(vec[11])
			#pic_video_str = '|'.join([vec[7], vec[9], vec[8], '%d' %videonum])
			#line = '\t'.join([vec[0], vec[1], vec[2], vec[24], pic_video_str])
			simhash = vec[9]
			videonum = int(vec[6]) + int(vec[7])
			pic_video_str = '|'.join([vec[3], vec[5], vec[4], '%d' %videonum])
			line = '\t'.join([vec[0], vec[1], vec[2], vec[8], pic_video_str])
			#if 'tagCategory_' in vec[8] and 'tagCategory_1004' not in vec[8]:
			#	print 'input list is:%s' %line
			bakline = line
			vec = line.strip().split('\t')

			if is_filter(vec):continue
			has_lev1_flag = has_lev1(vec)
			#if '抱歉，此微博已被删除。查看帮助' in vec[2] or '转发微博' in vec[2] or len(vec[2]) <= 9: continue
			if '抱歉，此微博已被删除。查看帮助' in vec[2] or len(vec[2]) <= 3 or ('转发微博' in vec[2] and len(vec[2]) <= 15): continue
			#print 'old tag list:%s' %(vec[3])

			#tmblog = mid_get_comment(vec[0])
			#print tmblog

			if vec[0] in midlist: continue
			if len(midlist) >500000:
				midlist.clear()

			#uid_tag_topn = []
			tagls = []
			taglist = ''
			cmmcnt = 10
			cmmflag = 0
			has_cmmt = False
			cmmt_flag = False
			has_lev2 = False
			vec2 = vec[2]
			while True:
				if ifs==tfs:
					#tmblog = mid_get_comment(vec[0], cmmcnt)
					tmblog = ' '.join(mid_get_comment(vec[0], cmmcnt)).strip()
					print '++++++++++comment is %s++++++++++++++' %tmblog
					cmmt_flag = True
					#uid_dict, tmblog = getHotComment(vec[0])
					#tmblog = ' '.join(tmblog)
					if len(tmblog) <= 15:
						has_cmmt = True
						break
					#uid_tag_topn = getUidClsTopN(uidInit, uid_dict, vec[1])

					vec[2] = '$#$&$#$&'.join([vec2, tmblog.encode('utf-8')])
				line = '\t'.join(vec)

				#print 'ori taglist:%s' %vec[3]
				tagls, has_lev2, pred_lev1, uid_lev1, is_adv, is_deep = decision(line, paramInit, uidAbility, cmmt_flag)
				print 'is_deep:', is_deep
				
				###add by lev3--->lev1()--2017-10-19
				taglist = '|'.join(tagls)
				if 'tagCategory_1004' in taglist:
					tagls = get_tags_tag(tagls)
				###add end

				if len(tagls) <= 0:
					tagls.append('1042015:tagCategory_1004@0.5')
				taglist = '|'.join(tagls)
				
				if 'tagCategory_1004' in taglist and (('$#$&$#$&' in vec[2] and len(vec[2])<=32) or ('$#$&$#$&' not in vec[2] and len(vec[2])<=24) or ('http' in vec[2] and len(vec[2])<=34) ):
					line = line+'_is_short_'
					print '---------------------content is very short--%s' %line
				
				if 'tagCategory_1004' in taglist and '__1' in ifs:
					#print 'file:%s cnt:%d' %(ifs, cmmflag)
					cmmcnt += 50
					cmmflag += 1
					if cmmflag >= 2: break
					continue
				break
			if cmmflag >= 2:
				print 'comment predict cnt >=2 mid:%s' %(vec[0])
				continue
			if has_cmmt:
				print 'comment len<=15 mid:%s' %(vec[0])
				continue
			if not has_lev1_flag and not has_lev2 and not is_deep:
				filter_flag = tag_filter(taglist, vec[2], pic_video_str)
				if filter_flag:
					print 'filter format--mid:%s, taglist:%s, is_pred:%d, is_uid_abi:%d' %(vec[0], taglist, pred_lev1, uid_lev1)
					continue

			is_lev1_filter = lev1_filter(tagls, paramInit, has_lev1_flag, has_lev2)
			ruku_cnt = 0
			if 'tagCategory_1004' not in taglist and not is_lev1_filter and (not has_lev1_flag or has_lev2):
				ruku_success = ruku_batch(vec[0], taglist)
				print 'ruku status:', ruku_success
				while not ruku_success:
					if ruku_cnt > 2: break
					ruku_success = ruku_batch(vec[0], taglist)
					ruku_cnt += 1
					time.sleep(0.01)
			
			if ruku_cnt > 2:
				print 'ruku cnt>2 failed, mid:%s, taglist:%s, is_pred:%d, is_uid_abi:%d' %(vec[0], taglist, pred_lev1, uid_lev1)
				continue

			ruku_cnt = 0
			if is_adv:
				taglist = '|'.join([taglist, '1042015:conceptTag_bacd6f4771952c9c5d254de71c485b05@0.666'])
				ruku_success = ruku_batch(vec[0], taglist)
				print 'ruku adv status:', ruku_success
				while not ruku_success:
					if ruku_cnt > 2: break
					ruku_success = ruku_batch(vec[0], taglist)
					ruku_cnt += 1
					time.sleep(0.01)

			if ruku_cnt > 2:
				print 'ruku adv cnt>2 failed, mid:%s, taglist:%s, is_pred:%d, is_uid_abi:%d' %(vec[0], taglist, pred_lev1, uid_lev1)
				continue
			#if 'tagCategory_1004' not in taglist and 'tagCategory_' in taglist and vec[0] not in midlist:
				#midlist[vec[0]] = 1
			
			###add write platform 20170914###
			#print 'write2platform %s, %s' %(vec[0],taglist)
			if 'tagCategory_1004' not in taglist:
				#print 'merge2platform begin!!!!+++++++'
				#print 'mid:', vec[0], 'taglist:', taglist
				merge2platform(vec[0],taglist)
				#print 'merge2platform end!!!!+++++++'

			if 'tagCategory_1004' not in taglist and 'tagCategory_' in taglist:
				print 'now tag list--mid:%s, taglist:%s, ori_pred:%d, is_pred:%d, is_uid_abi:%d' %(vec[0], taglist, has_lev1_flag, pred_lev1, uid_lev1)
				if vec[0] not in midlist:
					midlist[vec[0]] = 1
			#if 'tagCategory_1004' in taglist and '__1' not in ifs:
			#	print 'old line is:%s' %oldline

			vec[3] = taglist
			#vec[3] = '|'.join(user_tag)
			#ostr = '\t'.join(vec)
			adv_flag = '1'
			if is_adv:
				adv_flag = '0'
			deep_flag = '1'
			if not is_deep:
				deep_flag = '0'

				
			ostr = '\t'.join([vec[0], vec[3], simhash, adv_flag])

			#if ifs==tfs:#if indir == tmpdir:#说明是从临时目录--输出目录---备份目录
				#tmp--out--bak
			#elif tagCategory in taglist and tagCategory_1004 not in taglist:#打上标签路径:输入目录--输出目录---备份目录;
				#in--out--bak
			#else:没有打上标签:输入临时---临时目录
				#in--tmp
			if ifs == tfs and not is_lev1_filter:#tmp--out--bak(打上标签的入库)
				#if 'tagCategory_1004' not in taglist and 'tagCategory' in taglist:
				ofs.write(ostr + '\n')
				ofs.flush()
				#os.fsync(ofs)
				bfs.write(line + '\n')
			#elif 'tagCategory_1004' not in taglist and 'tagCategory' in taglist:#in--out--bak
			elif ifs != tfs and not is_lev1_filter:#in--out--bak(打上标签的入库)
				#if not is_lev1_filter:
				ofs.write(ostr + '\n')
				ofs.flush()
				#os.fsync(ofs)
				bfs.write(line + '\n')
				#print 'write file ofs:%s, bfs:%s' %(ofs, bfs)
			elif 'tagCategory_1004' in taglist and '__1' not in ifs:#in--tmp(in--out--bak中没有打上标签的进入补打流程)
				tfsx.write(oldline.strip()+'$$$'+adv_flag+'\n')
				tfsx.flush()
				#os.fsync(tfsx)
			#if okcn % 100 == 0:
			#	time.sleep(0.2)
			okcn += 1
		except Exception,e:
			print "line error:", e
			print '\nline is:%s' %bakline
			#continue
	ofs.close()
	bfs.close()
	if len(tfs) > 0 and '__1' not in ifs:
		tfsx.close()
	iterval = time.time() - begin
	print 'rows:%d, spent:%d' %(nrow, iterval) 
	#time.sleep(0.2)
	#print 'close ofs, bfs, tfsx handle'
	#print 'close ifs:%s, ofs:%s, bfs:%s, tfs:%s, tfsx:%s' %(ifs, ofs, bfs, tfs, tfsx)

def consumer(midlist, queue, paramInit, uidInit, uidAbility, idir, odir, bdir, tdir):
	#def run(self):
		while True:
			ifile = queue.get()
			#print 'queue get file:%s' %(ifile)
			ifs = os.path.join(idir, ifile)
			if not os.path.exists(ifs):continue

			ofs = os.path.join(odir, ifile + '_1')#ofs-->ofs_1
			bfs = os.path.join(bdir, ifile)
			tfs = os.path.join(tdir, ifile)
			tfsx = tfs + '__1'
			#processing the item
			try:
				processFile(midlist, paramInit, uidInit, uidAbility, ifs, ofs, bfs, tfs, tfsx)
			except Exception,e:
				print "error is:", Exception, ":", e
			cmd = 'rm -rf %s' %(ifs)
			os.system(cmd)
			#self.queue.task_done()
		#self.queue.task_done()
		return

def product(dicts, que, time_interval, idir):
	#def run(self):
		while True:
			for ifile in os.listdir(idir):
				if ifile == '.' or ifile == '..' or ifile.startswith('.'): continue
				ifs = os.path.join(idir, ifile)
				if not os.path.exists(ifs): continue
				#if not os.path.getsize(ifs) or (ifile.count('drop') >= 2 or ('txt' in ifile and 'drop' in ifile)):
				if not os.path.getsize(ifs):# or '_1' in ifile:#add __1 and _1 judge
					#print 'ifs:%s size is 0 and delete' %ifs
					cmd = 'rm -rf %s' %(ifs)
					os.system(cmd)
					continue

				statinfo = os.stat(ifs)
				delta = int(time.time()) - int(statinfo.st_mtime)
				if delta >= int(time_interval) and ifile not in dicts:
					#print 'ifs:%s delta is :%d, need process' %(ifs, delta)
					que.put(ifile)
					#print 'que put file:%s' %(ifile)
					if len(dicts) >= 200000: dicts.clear()
					dicts[ifile] = 1
			time.sleep(5)

def main():
	if len(sys.argv) != 37:#add uid ability list, online time interval,  offline time interval, indir, tmpdir, outdir, bakdir, online queue len, offline queue len, online process thread num, offline process thread num
		print '[exec_name, model_file, vocabulary_list, classname_feature, classname_type, classname_objid, classname_rootid, model_adv_nb, model_adv_lr, vocabulary_list_adv, online_class_list, qiyi.online.lev3, type2root_file, id2root_file, word2root_file]'
		exit(-1)

	paramInit = SysInit(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.  argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14], sys.argv[15], sys.argv[16], sys.argv[17], sys.argv[33], sys.argv[34], sys.argv[35], sys.argv[36])
	uidInit = UidInit(sys.argv[18])
	uidAbility = pnd(sys.argv[30], sys.argv[31], sys.argv[32])

	#start thread for tag in time and on time in 30minites
	#while True:#kafka input stream or filedir sys.argv[16]
	#uid ability score	 ---sys.argv[15]
	#online time interval---sys.argv[16](default 1min)
	#offlinetime interval---sys.argv[17](default 30min)
	#input dir ---sys.argv[18]
	#tempt dir ---sys.argv[19]
	#output dir---sys.argv[20]
	#bak 	dir---sys.argv[21]
	#online queue len---sys.argv[22]
	#offline queue len---sys.argv[23]
	#online process thread num---sys.argv[24]
	#offline process thread num---sys.argv[25]

	print 'new product and consum queue!!'
	#que1 = Queue.Queue(maxsize = int(sys.argv[25]))
	#que2 = Queue.Queue(maxsize = int(sys.argv[26]))
	que1 = Queue(maxsize = int(sys.argv[25]))
	que2 = Queue(maxsize = int(sys.argv[26]))
	dictFile = {}

	is_comment = int(sys.argv[29])
	if is_comment: pass
	'''
	print 'weibo login!!!'
	#account='yanlei54@163.com'
	#passwd='yL523523'
	account='867743442@qq.com'
	passwd='123456@'
	weiboLogin = WeiboLogin(account, passwd)#账号和密码
	if weiboLogin.Login() == True:
		print "login successed!"
	else:
		print 'login unsuccessed!'
		sys.exit(-1)
	'''

	print 'start product and consum queue!!'
	thread_product = [threading.Thread(target = product, args = (dictFile, que1, sys.argv[19], sys.argv[21])), threading.Thread(target = product, args = (dictFile, que2, sys.argv[20], sys.argv[22]))]
	for ithread in thread_product:
		ithread.start()

	midlist = {}
	thread_consumer1 = [threading.Thread(target = consumer, args = (midlist, que1, paramInit, uidInit, uidAbility, sys.argv[21], sys.argv[23], sys.argv[24], sys.argv[22])) for i in range(int(sys.argv[27]))]
	for ithread in thread_consumer1:
		ithread.start()

	thread_consumer2 = [threading.Thread(target = consumer, args = (midlist, que2, paramInit, uidInit, uidAbility, sys.argv[22], sys.argv[23], sys.argv[24], sys.argv[22])) for i in range(int(sys.argv[28]))]#offline time interval, tempt dir, output dir, bak dir, tmp dir
	for ithread in thread_consumer2:
		ithread.start()
	print 'start product and consum queue ok!!!'

if __name__ == '__main__':
	main()
