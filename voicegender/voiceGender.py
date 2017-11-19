#coding=utf-8
'''
http://localhost:12345/params?content=12\tabc
http://localhost:12345/params?content=2\t3
'''

from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn
import sys
import urllib
from sklearn.externals import joblib
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
import pandas as pd
#import tensorflow as tf

#from utils import TextLoader
#from model import Model
#csv.field_size_limit(sys.maxsize)

#queue = Queue.Queue(10)
#models = joblib.load('../model/model_lr')

#xumm change:
import xgboost as xgb
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
import shutil, json
import simplejson as json


class TestHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        #print('self.path:', self.path)

        if '?' in self.path:  
            mpath,margs = urllib.splitquery(self.path)  
            #print('mpath:', mpath)
            #print('margs', margs)
            retdic = {}

            content = margs.split('=')
            if len(content) <= 1:
                print 'margs:%s' %margs
                retdic['code'] = 1
                retdic['msg'] = 'External ERROR:no param is transfromed'
                retdic['data'] = 'none'
            
            else:
                file_name = content[1]
                fname = file_name.split('.')[0]+'.wav'
                if file_name not in os.listdir('data'):
                    retdic['code'] = 2
                    retdic['msg'] = 'Internal ERROR:rsync mp3 not successful'
                    retdic['data'] = 'unknown'
                else:
                    #print('content', content)
                    #mid, weibo = content[1].split('\\t')
                    #self.wfile.write('file_name:%s' % file_name)
                    indir_path = os.path.join('data', file_name)
                    outdir_path = os.path.join('data', fname)
                    if not os.path.exists(outdir_path):
                        flag = False
                        cmd = 'ffmpeg -i %s -acodec pcm_s16le -ac 1 -ar 8000 -vn %s' %(indir_path, outdir_path)
                        ret= os.system(cmd)
                        if not ret: flag = True
                        retdic['code'] = 3
                        retdic['msg'] = 'Internal ERROR:ffmpeg transform error'
                        retdic['data'] = 'unknown'
                            
                        #ret= os.system(cmd)
                        #if ret:
                        #    while ret:
                        #        ret= os.system(cmd)
                        #        time.sleep(0.01)
                    if os.path.exists(outdir_path) or flag:
                        data_read = robjects.r.processFolder(fname)
                        data_read = pandas2ri.ri2py(data_read)

                        result = predict(xgboost_model, data_read)#.encode('utf8'))
                    
                        retdic['code'] = 0
                        retdic['msg'] = 'success'
                        retdic['data'] = result

            self.protocal_version = 'HTTP/1.1'
            self.send_response(200)
            encoding = sys.getfilesystemencoding()
            self.send_header("Content-type", "text/html; charset=%s" % encoding)
            self.end_headers()
            #content = result
            #self.wfile.write('file_name:%s' % file_name)
            #self.wfile.write('\n')
            self.wfile.write(json.dumps(retdic))
            #self.wfile.write(content)
            #self.wfile.write('Predict Result:%s' % result[0])
            #self.wfile.write(content)

class ThreadingHTTPServer(HTTPServer, ThreadingMixIn):
    pass

def start_server(port):
    addr = '10.77.6.239'
    http_server = ThreadingHTTPServer((addr, int(port)), TestHTTPHandler)
    print ("HTTP server is at: http://%s:%s/" % (addr, port))
    http_server.serve_forever()

#xumm change
'''
def transform(text, seq_length, vocab):
    x = map(vocab.get, text)
    x = map(lambda i: i if i else 0, x)
    if len(x) >= seq_length:
        x = x[:seq_length]
    else:
        x = x + [0] * (seq_length - len(x))
    return x
'''

#xumm change
def predict(model, data_read):
    pred = model.predict(xgb.DMatrix(data_read), ntree_limit=model.best_ntree_limit)
    
    if pred < 0.5:
        sex = 'male'
    else:
        sex = 'female' 

   #names = os.listdir('data')
   #file_read_path = 'data/'
   #file_save_path = 'data_save/'
    #for name in names:
        #file_read = file_read_path + name
        #file_save = file_save_path + name
        #shutil.copyfile(file_read, file_save)
        #os.remove(file_read)

    return sex
    

def loadModel():
    global xgboost_model
    model_save = open('model.pkl', 'rb')
    xgboost_model = pickle.load(model_save)
    model_save.close()



def init():
   
    pandas2ri.activate()
    robjects.r.source('feature_extract.R')
    loadModel()

def main():
    init()
    print ('Initialization finished!')
    start_server('8265')
    
if __name__ == '__main__':  
    main()
