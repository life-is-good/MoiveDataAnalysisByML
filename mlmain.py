#coding=utf-8
import storesentimentclassifier as ssc # 分类器类
import posnegmlfeature as pnf # 预测类
import textprocessing as tp # 处理excel或者txt类
import pickle 
from random import shuffle
from nltk.classify.scikitlearn import SklearnClassifier
import os
import time

if __name__ == "__main__":
    #三部电影《乘风破浪》《长城》《西游伏魔篇》，分别跑一遍，然后将三部电影合起来再跑一遍
    #使用MutiNB,LogisticRegression,SVM三种分类器
    
    print '开始训练分类器'
    # 1. Load positive and negative review data
    path = os.getcwd()
    print '当前路径'+path
    start_time = time.time()
    print start_time
    pos_review = tp.seg_fil_senti_excel(path+"\\seniment review set\\THREEMIXPOS.xls", 1, 1)
    neg_review = tp.seg_fil_senti_excel(path+"\\seniment review set\\THREEMIXNEG.xls", 1, 1)
    test_review = test_review = tp.seg_fil_senti_excel(path+"\\seniment review set\\THREEMIXTEST.xls", 1, 1)
    
    pos = pos_review
    neg = neg_review
    
    # 2. Feature extraction function
    # Choose word_scores extaction methods
    #word_scores = create_word_scores()
    #word_scores = create_bigram_scores()
    word_scores = ssc.create_word_bigram_scores(pos,neg)
    
    # 3. Transform review to features by setting labels to words in review
    best_words = ssc.find_best_words(word_scores, 1500) # Set dimension and initiallize most informative words

    # posFeatures = ssc.pos_features(ssc.bigrams)
    # negFeatures = ssc.neg_features(ssc.bigrams)
    
    # posFeatures = ssc.pos_features(ssc.bigram_words)
    # negFeatures = ssc.neg_features(ssc.bigram_words)
    
    # posFeatures = ssc.pos_features(ssc.best_word_features)
    # negFeatures = ssc.neg_features(ssc.best_word_features)
    
    posFeatures = ssc.pos_features(pos,ssc.best_word_features_com,best_words)
    negFeatures = ssc.neg_features(neg,ssc.best_word_features_com,best_words)
    
    # 4. Train classifier and examing classify accuracy
    # Make the feature set ramdon
    shuffle(posFeatures)
    shuffle(negFeatures)
    
    # 5. After finding the best classifier,store it and then check different dimension classification accuracy
    # 75% of features used as training set (in fact, it have a better way by using cross validation function)
    size_pos = int(len(pos_review) * 0.75)
    size_neg = int(len(neg_review) * 0.75)
    
    train_set = posFeatures[:size_pos] + negFeatures[:size_neg]
    test_set = posFeatures[size_pos:] + negFeatures[size_neg:]
    
    test, tag_test = zip(*test_set)
    
    classifier = []
    classifier = ssc.cal_classifier_accuracy(train_set,test,tag_test)
    #选择准确度最高的分类器,正常情况下是选择准确性最高的
#     object = ssc.find_score_max(classifier)
    #分别测试多项分布贝叶斯，逻辑回归，svm比较性能
#     object = classifier[1][0]  #多项分布
    object = classifier[2][0]  #逻辑回归
#     object = classifier[4][0]   #svm
    print '选择的分类器是：'
    print object
    # 存储分类器
#     ssc.store_classifier(object, train_set, path)
    print '存储分类器'
    object_classifier = SklearnClassifier(object)
    object_classifier.train(train_set)
    pickle.dump(object_classifier, open(path+'/great_LRclassifier.pkl','w'))
    print '结束训练分类器'
    print '开始预测'
#     test_word_scores = pnf.create_words_bigrams_scores()
#     test_best_words = pnf.find_best_words(test_word_scores, 1500)
    clf = pickle.load(open(path+'/great_LRclassifier.pkl'))
    pred = clf.prob_classify_many(pnf.extract_features(test_review,best_words))
    print '存储预测结果'
    pnf.store_predict_result(path,pred)
    print '结束预测'
    end_time = time.time()
    print end_time
    print end_time-start_time
    
    
    
    
    
    
    
    
    
    
