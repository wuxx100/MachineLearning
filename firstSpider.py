#!/usr/bin/python
#  -*- coding:utf-8 -*-

import requests ##实现爬虫
from bs4 import BeautifulSoup   ##解析网页html文件，比正则表达式简单
import re   ##重整字符串，筛选
import os

from requests.packages.urllib3.exceptions import InsecureRequestWarning ##去掉warning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(UserWarning)


print '-'*100
outpath='./downloadFile/'
if not os.path.exists(outpath):
    print 'no file'
    os.mkdir(outpath)
url='https://serenad-public.cnes.fr/Niveau0/SERENAD0/FROM_NTMF/MSG'
print 'Initing, download path:', outpath

print '-'*100
print 'Reading:',url
r = requests.get(url, verify=False).text
soup = BeautifulSoup(r)

#soup.p p是段落
#soup.a a是超链接
print '-'*100
print 'Processing:',url
for name in soup.findAll('a',href=True):

    ## now we are in /MSG
    subUrl = 'https://serenad-public.cnes.fr'+name['href'] #like MSG/2017
    unUsedWord=re.compile(r'delete')
    usefulWord=re.compile(r'MSG')
    matchUnUsedWord = re.search(unUsedWord , subUrl)
    matchUsefulWord = re.search(usefulWord , subUrl)

    if(matchUsefulWord and not matchUnUsedWord):
        subR=requests.get(subUrl, verify=False).text
        subSoup=BeautifulSoup(subR)

        print '-' * 100
        print 'sub Processing:', subUrl

        for subName in subSoup.findAll('a',href=True):

            ## now we are in /MSG/* (like /MSG/2017)
            subSubUrl='https://serenad-public.cnes.fr'+subName['href'] #like MSG/2017/100
            subUsefulWord=re.compile(r'MSG/2017')
            subMatchUnUsedWord = re.search(unUsedWord, subSubUrl)
            subMatchUsefulWord = re.search(subUsefulWord, subSubUrl)
            if (subMatchUsefulWord and not subMatchUnUsedWord):
                subSubR=requests.get(subSubUrl, verify=False).text
                subSubSoup=BeautifulSoup(subSubR)

                print '-' * 100
                print 'sub Processing:', subSubUrl

                for subSubName in subSubSoup.findAll('a',href=True):

                    ##now we are in /MSG/*/* (like /MSG/2017/001)
                    subSubSubUrl='https://serenad-public.cnes.fr'+subSubName['href'] #like MSG/2017/100/G1380160.17b.Z
                    subSubUsefulWord = re.compile(r'M138')
                    subSubMatchUnUsedWord = re.search(unUsedWord, subSubSubUrl)
                    subSubMatchUsefulWord = re.search(subSubUsefulWord, subSubSubUrl)

                    if (subSubMatchUsefulWord and not subSubMatchUnUsedWord):
                        print '-' * 100
                        print 'downloading:', subSubSubUrl
                        splitedWords = subSubSubUrl.split('/')
                        outfname = outpath + splitedWords[-3] + '_' + splitedWords[-2] + '_' + splitedWords[-1]
                        print outfname
                        # r = requests.get(subSubSubUrl, verify=False, stream=True)
                        # if (r.status_code == requests.codes.ok):
                        #     fsize = int(r.headers['content-length'])
                        #     print 'Downloading %s (%sMb)' % (outfname, fsize / (1024 * 1024))
                        #     with open(outfname, 'wb') as filedown:
                        #         for chunk in r.iter_content(chunk_size=1024):
                        #             if chunk:
                        #                 filedown.write(chunk)
                        #         filedown.close()







