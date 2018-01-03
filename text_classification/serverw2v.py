#!/usr/bin/env python
# encoding: utf-8
#coding=utf-8
'''
http://localhost:12345/params?content=12\tabc
http://localhost:12345/params?content=2\t3
'''

from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn
import sys
import urllib
#from sklearn.externals import joblib
import threading
import Queue
import random  
import time  
import os, sys
import time
import csv
import argparse
import cPickle as pickle

import numpy as np
#import pandas as pd
#import tensorflow as tf

#from utils import TextLoader
#from model import Model
csv.field_size_limit(sys.maxsize)

#queue = Queue.Queue(10)
#models = joblib.load('../model/model_lr')

#xumm change:
import re
import jieba
import json
from process import ProcessText
from pyfasttext import FastText
from content_test import ContCmp



class TestHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        #print 'self.path:', self.path

        if '?' in self.path:  
            mpath,margs = urllib.splitquery(self.path)  
            #print('mpath:', mpath)
            #print('margs', margs)

            content = margs.split('=')
            #print 'content', content
            #mid, weibo = content[1].split('\\t')

            weibo = content[-1]
	    weibo = urllib.unquote(weibo)
            result = predict(fasttext_model, processtext, weibo)#.encode('utf8'))
        
            self.protocal_version = 'HTTP/1.1'
            self.send_response(200)
            encoding = sys.getfilesystemencoding()
            self.send_header("Content-type", "text/html; charset=%s" % encoding)
            self.end_headers()
            content = result
	    
            #self.wfile.write('weibo_fenci:%s' % weibo)
            self.wfile.write('Predict Result:%s' % content)
            #self.wfile.write('Predict Result:%s' % result[0])
            #self.wfile.write(content)

class ThreadingHTTPServer(HTTPServer, ThreadingMixIn):
    pass

def start_server():
    addr = '10.77.6.241'
    port = 8188
    http_server = ThreadingHTTPServer((addr, port), TestHTTPHandler)
    print ("HTTP server is at: http://%s:%s/" % (addr, port))
    http_server.serve_forever()

#xumm change
def predict(model, fenci, weibo):
    weibo_join = fenci.process(weibo)
    words = ''.join(weibo_join.split(' '))
    weibo_content = weibo_join + '\n'
    label, prob = model.predict_proba_single(weibo_content, k = 1)[0] 
    
    label_other_form = '1042015:' + label
    #flag, _ = contcmp.check_is_exist(label_other_form, words)
    flag = False
    if label_other_form in labels_list and prob > 0.8:
        if 'tagCategory_046' in label_other_form:
            return '@'.join([label_other_form, str(prob)])
        flag, kcnt = contcmp.check_is_exist(label_other_form, words)
        if not flag or ('tagCategory_060' in label_other_form and kcnt < 2):
            return  "1042015:tagCategory_1004@0.5"
        else:
            return '@'.join([label_other_form, '0.6111'])
    else:
        return "1042015:tagCategory_1004@0.5"
    #predict_result = '@'.join([label_other_form, str(prob)])

    #return  predict_result
    

def loadModel():
    global fasttext_model
    fasttext_model = FastText()
    fasttext_model.load_model('3Ngram_3mincount_1wminlabel.bin')




def init():
    global processtext
    processtext = ProcessText()
    
    global labels_list
    with open("both_labels.pkl", "rb") as f:
        labels_list = pickle.load(f)
    
    global contcmp
    contcmp = ContCmp("root_feature_file.allid")
    #loadModel()
    
    global fasttext_model
    fasttext_model = FastText()
    fasttext_model.load_model('3Ngram_3mincount_1wminlabel.bin')

def main():
    init()
    print ('Initialization finished!')
    start_server()
    
if __name__ == '__main__':  
    main()
