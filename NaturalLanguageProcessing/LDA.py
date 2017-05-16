# !/usr/bin/python
# -*- coding:utf-8 -*-

## http://www.cnblogs.com/bentuwuying/p/6219970.html 讲解LSA与pLSA

from gensim import corpora, models, similarities

if __name__ == '__main__':
    f = open('LDA_test.txt')
    stop_list = set('for a of the and to in'.split())       ##停止词是在一个集合里
    print 'After'
    texts = [[word for word in line.strip().lower().split() if word not in stop_list] for line in f]
    ##strip是为了去回车等，lower为了小写
    print 'Text = ', texts

    dictionary = corpora.Dictionary(texts)      #corpora 语料，是个写好的包，在这里为了生成词典
    print dictionary
    corpus = [dictionary.doc2bow(text) for text in texts]   #把文档变成tf向量(corpus是每个文档中每个词出现的频数)
    corpus_tfidf = models.TfidfModel(corpus)[corpus]        #用corpus生成tfidf


    # for i in corpus_tfidf:
    #     print i

    topicNum=2

    print '\nLSI Model:'
    lsi = models.LsiModel(corpus_tfidf, num_topics=topicNum, id2word=dictionary)       #id2word是为了输出时输出词，而不是词的编号
    topic_result = [topic for topic in lsi[corpus_tfidf]]       ##lsi[corpus_tfidf]是指每篇文档是什么模型(D_T)
    print u'文档——主题'
    print topic_result                                      ##矩阵分解可能结果是负数，为了避免使用NMS非负矩阵分解（虽说我不知道什么是nms...）
    print 'LSI Topics:'
    print u'主题——词'
    print lsi.print_topics(num_topics=topicNum, num_words=5)
    similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])  # 任意两个文档间的相似度
    print 'Similarity:'
    print list(similarity)

    print '\nLDA Model:'
    lda = models.LdaModel(corpus_tfidf, num_topics=topicNum, id2word=dictionary,
                          alpha='auto', eta='auto', minimum_probability=0.001, passes=10)   #alpha主题超参数, eta词超参数, passes循环多少回
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
    print u'文档——主题'
    print doc_topic
    # for doc_topic in lda.get_document_topics(corpus_tfidf):
    #     print doc_topic
    print u'主题——词'
    for topic_id in range(topicNum):
        print 'Topic', topic_id
        print lda.show_topic(topic_id)
    similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
    print 'Similarity:'
    print list(similarity)

    ##可以看出来LDA不适合短文档