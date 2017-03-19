#coding=utf-8
import textprocessing as tp
import pickle
import itertools
from random import shuffle
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import sklearn
import os

# 1. 加载测试集
# path = os.getcwd()
# test_review = tp.seg_fil_senti_excel(path+"/seniment review set/GREATEWALLTEST.xls", 1, 6)

# 2. 提取特征
# Used for transform review to features, so it can calculate sentiment probability by classifier
# def create_words_bigrams_scores():
#     posdata = tp.seg_fil_senti_excel(path+"/seniment review set/GREATEWALLPOS.xls", 1, 6)
#     negdata = tp.seg_fil_senti_excel(path+"/seniment review set/GREATEWALLNEG.xls", 1, 6)
#     
#     posWords = list(itertools.chain(*posdata))
#     negWords = list(itertools.chain(*negdata))
# 
#     bigram_finder_pos = BigramCollocationFinder.from_words(posWords)
#     bigram_finder_neg = BigramCollocationFinder.from_words(negWords)
#     posBigrams = bigram_finder_pos.nbest(BigramAssocMeasures.chi_sq, 5000)
#     negBigrams = bigram_finder_neg.nbest(BigramAssocMeasures.chi_sq, 5000)
# 
#     pos = posWords + posBigrams
#     neg = negWords + negBigrams
# 
#     word_fd = FreqDist()
#     cond_word_fd = ConditionalFreqDist()
#     for word in pos:
#         word_fd[word] += 1
#         cond_word_fd['pos'][word] += 1
#     for word in neg:
#         word_fd[word] += 1
#         cond_word_fd['neg'][word] += 1
# 
#     pos_word_count = cond_word_fd['pos'].N()
#     neg_word_count = cond_word_fd['neg'].N()
#     total_word_count = pos_word_count + neg_word_count
# 
#     word_scores = {}
#     for word, freq in word_fd.iteritems():
#         pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
#         neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
#         word_scores[word] = pos_score + neg_score
# 
#     return word_scores
# 
# def find_best_words(word_scores, number):
#     best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
#     best_words = set([w for w, s in best_vals])
#     return best_words

# Initiallize word's information score and extracting most informative words
# 计算每个词的信息熵，提取信息量最大的词作为特征词
# word_scores = create_words_bigrams_scores()
# best_words = find_best_words(word_scores, 1500) # Be aware of the dimentions

def best_word_features(words,best_words):
    return dict([(word, True) for word in words if word in best_words])

def best_word_features_com(words,best_words):
    
    d1 = dict([(word, True) for word in words if word in best_words])
    d2 = dict([(word, True) for word in nltk.bigrams(words) if word in best_words])
    d3 = dict(d1, **d2)
    return d3

# 3. 测试集提取特征
def extract_features(dataset,best_words):
    feat = []
    for i in dataset:
#         feat.append(best_word_features(i,best_words)) 
        feat.append(best_word_features_com(i,best_words))
    return feat

# 4. 加载分类器
# clf = pickle.load(open(path+'/classifier.pkl'))

# 计算测试集的概率
# pred = clf.prob_classify_many(extract_features(test_review))

#保存
def store_predict_result(path,pred):
    p_file = open(path+'/result/greate_LRfinal.txt', 'w')
    for i in pred:
        p_file.write(str(i.prob('pos')) + ' ' + str(i.prob('neg')) + '\n')
    p_file.close()
    
    pred2 = []
    pos_count=0
    neg_count=0
    for i in pred:
        pred2.append([i.prob('pos'), i.prob('neg')])
        if i.prob('pos')>i.prob('neg'):
            pos_count += 1
        else:
            neg_count += 1
    
    print '好评占：','%f'%((pos_count*1.0)/((pos_count+neg_count)*1.0))
    print '差评占：','%f'%((neg_count*1.0)/((pos_count+neg_count)*1.0))