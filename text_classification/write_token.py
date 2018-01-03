#!/usr/bin/env python
# encoding: utf-8

import sys
import json
import hashlib
import hmac
import hashlib
import urllib2
import urllib,redis

db=redis.Redis("rm20435.hebe.grid.sina.com.cn",20435,db="3")
tauth2_token=db.get("tauth2_token_1586188222")

def httpCall(command):
	return commands.getstatusoutput(command)[1]

def hmacSignature(tauth_token_secret, param_str):
	h = hmac.new(tauth_token_secret, param_str, hashlib.sha1)
	s = h.digest()
	return s.encode('base64').rstrip()

def createAuthHeader(token, param, sign):
	m={'token':token,'param':param, 'sign':sign}
	arr=urllib.urlencode(m).split('&')
	#authorizationHeader="Authorization:TAuth2 "
	authorizationHeader="TAuth2 "
	for value in arr:
		pair=value.split("=")
		authorizationHeader+=pair[0] + "=\"" + pair[1] + "\","
	return authorizationHeader[0: len(authorizationHeader) - 1]

def get_header():
	global tauth2_token
	global db

	header=""
	for i in range(3):
		try:
			results=json.loads(tauth2_token.strip('\n').strip('\r'))
			token=results['tauth_token'].encode('utf8')
			token_secert=results['tauth_token_secret'].encode('utf8')
			user="2768965585"
			sign=hmacSignature(token_secert, "uid=" + user)
			header=createAuthHeader(token,"uid="+user,sign)
			return header
		except Exception as e:
			tauth2_token=db.get("tauth2_token_1586188222")
			print 'get_header error',e
	return header

if __name__=="__main__":
	url="http://i.api.weibo.com/2/uattr/get_attr.json?source=2936099636&types=level&uids=2768965585"
	header = get_header()
	req = urllib2.Request(url,None,{"Authorization":header})
	jsonstr = urllib2.urlopen(req,timeout=5).read()
	res = json.loads(jsonstr)
