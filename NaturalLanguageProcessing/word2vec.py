# etc/bin/python
# -*- encoding: utf-8 -*-

from time import time
from gensim.models import Word2Vec
import sys
import os


reload(sys)
sys.setdefaultencoding('utf-8')


class LoadCorpora(object):
    def __init__(self, s):
        self.path = s

    def __iter__(self):
        f = open(self.path,'r')
        for line in f:
            yield line.split(' ')




if __name__ == '__main__':
    if not os.path.exists('news.model'):
        sentences = LoadCorpora('news.dat')
        model = Word2Vec(sentences, size=200, min_count=5)  # 词向量维度为200，丢弃出现次数少于5次的词
        model.save('news.model')


    W2Vmodel = Word2Vec.load('news.model')
    # print u'词典中词的个数：', len(model.wv.vocab)
    # for i, word in enumerate(model.wv.vocab):
    #     print word,
    #     if i % 25 == 24:
    #         print
    # print

    intrested_words = ('女朋友', '手机')
    print '特征向量：'
    for word in intrested_words:
        print word, W2Vmodel[word]
    for word in intrested_words:
        result = W2Vmodel.most_similar(word)
        print '与', word, '最相近的词：'
        for w, s in result:
            print '\t', w, s

    words = ('女神', '网红', '女朋友')
    for i in range(len(words)):
        w1 = words[i]
        for j in range(i+1, len(words)):
            w2 = words[j]
            print '%s 和 %s 的相似度为：%.6f' % (w1, w2, W2Vmodel.similarity(w1, w2))

    print '========================'
    opposites = (
                (['中国', '北京'], ['美国']),
                (['男', '工作'], ['女'])
                )
    for positive, negative in opposites:
        result = W2Vmodel.most_similar(positive=positive, negative=negative)
        for word in positive:
            print word,
        print '-',
        for word in negative:
            print word,
        print '：'
        for word, similar in result:
            print '\t', word, similar

    print '========================'
    words_list = ('苹果 三星 美的 海尔', '中国 日本 韩国 美国 法国')
    for words in words_list:
        print words, '离群词：', W2Vmodel.doesnt_match(words.split(' '))
