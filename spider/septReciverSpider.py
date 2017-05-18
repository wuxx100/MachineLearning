#!/usr/bin/python
#  -*- coding:utf-8 -*-

import urllib2 ##实现爬虫
from urllib2 import URLError
import requests
from bs4 import BeautifulSoup   ##解析网页html文件，比正则表达式简单
import re   ##重整字符串，筛选
import os


print '-'*100
outpath='./fileSeptReciver/'
if not os.path.exists(outpath):
    print 'no file'
    os.mkdir(outpath)

usefulStation=['ABPO','AGGO','BJNM','BRUX','CEBR','CEDU','CHPI','COCO','DARW','DAV1','DUBO','EPRT',
               'FAA1',	'FALK',	'GAMG',	'HERS',	'HOB2',	'KAT1',	'KIRU',	'KOS1',	'KOUR',	'MAC1',	'MAJU',
               'MAL2','MAS1','MGUE','MOBS','NAUR','NNOR','PICL','REDU','ROAP','STJ3','STR1','SYDN','TID1',
               'TLSG','TOW2','TWTF','UNX3','USN8','VILL','WES2','WTZS','YAR2','YEL2','ZWE2'
               ]
i = 0
for year in range(14,17,1):
    suboutpath=outpath+str(year)+'/'
    if not os.path.exists(suboutpath):
        print 'no file'
        os.mkdir(suboutpath)

    for dayIdx in range(1,367,1):
        subsuboutpath = suboutpath + str(dayIdx) + '/'
        if not os.path.exists(subsuboutpath):
            print 'no file'
            os.mkdir(subsuboutpath)

        for station in usefulStation:
            day ="%03d" % dayIdx
            fileName=station.lower()+str(day)+'0.'+str(year)+'n.Z'
            print fileName
            url='ftp://cddis.gsfc.nasa.gov/gnss/data/daily/20'+str(year)+'/'+str(day)+'/'+str(year)+'n/'+fileName
            print url

            try:
                f=urllib2.urlopen(url)
            except URLError, e:
                print e
            else:
                print f
                print i
                i+=1
                # splitedWords = url.split('/')
                outfname = subsuboutpath +fileName
                print 'Downloading', outfname
                with open(outfname, 'wb') as filedown:
                    filedown.write(f.read())

            print '-'*100


