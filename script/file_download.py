import urllib.request
import os
import re
#os.chdir('D:\\voxforge speech files\\')
#os.chdir('D:\\voxforge speech files\\')  # 改变当前路径
# refiles=open('speech_files_path.txt','w+')#存储所有下载连接
mainpath = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'


def gettgz(url):
    page = urllib.request.urlopen(url)
    html = page.read()
    html = str(html, encoding="utf-8")


    reg = r'href=".*\.tgz"'
    tgzre = re.compile(reg)
    tgzlist = re.findall(tgzre, html)  # 找到所有.tgz文件
    for i in tgzlist:
        filename = i.replace('href="', '')
        filename = filename.replace('"''"', '')
        filename = filename[:-1]
        print('正在下载：' + filename)  # 提示正在下载的文件
        downfile = i.replace('href="', mainpath)
        downfile = downfile.replace('"', '')  # 得到每个文件的完整连接
        req = urllib.request.Request(downfile)  # 下载文件
        ur = urllib.request.urlopen(req).read()
        
        f = open(filename, 'wb')
        f.write(ur)
        f.close()




html = gettgz(mainpath)
# refiles.close()
