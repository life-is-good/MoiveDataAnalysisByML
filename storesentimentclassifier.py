#coding=utf-8
import pickle
import itertools
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import sklearn
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import accuracy_score

# Feature extraction function
# 1 Use all words as features
def bag_of_words(words):
    return dict([(word, True) for word in words])

# 2 Use bigrams as features (use chi square chose top 200 bigrams)
def bigrams(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(bigrams)

# 3 Use words and bigrams as features (use chi square chose top 200 bigrams)
def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)


# 4 Use chi_sq to find most informative features of the review
# 4.1 First we should compute words or bigrams information score
def create_word_scores(posdata,negdata):
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in negWords:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores
#4.2
def create_bigram_scores(posdata,negdata):
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder_pos = BigramCollocationFinder.from_words(posWords)
    bigram_finder_neg = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder_pos.nbest(BigramAssocMeasures.chi_sq, 8000)
    negBigrams = bigram_finder_neg.nbest(BigramAssocMeasures.chi_sq, 8000)

    pos = posBigrams
    neg = negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1

    for word in neg:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1        

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

# 4.3Combine words and bigrams and compute words and bigrams information scores
def create_word_bigram_scores(posdata,negdata):
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder_pos = BigramCollocationFinder.from_words(posWords)
    bigram_finder_neg = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder_pos.nbest(BigramAssocMeasures.chi_sq, 5000)
    negBigrams = bigram_finder_neg.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = posWords + posBigrams
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in neg:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1
        

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

# 5 Second we should extact the most informative words or bigrams based on the information score
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

# 6 Third we could use the most informative words and bigrams as machine learning features
# Use chi_sq to find most informative words of the review
def best_word_features(words,best_words):
    return dict([(word, True) for word in words if word in best_words])

# Use chi_sq to find most informative bigrams of the review
def best_word_features_bi(words,best_words):
    return dict([(word, True) for word in nltk.bigrams(words) if word in best_words])

# Use chi_sq to find most informative words and bigrams of the review
def best_word_features_com(words,best_words):
    d1 = dict([(word, True) for word in words if word in best_words])
    d2 = dict([(word, True) for word in nltk.bigrams(words) if word in best_words])
    d3 = dict(d1, **d2)
    return d3

#7 Transform review to features by setting labels to words in review
def pos_features(pos,feature_extraction_method,best_words):
    posFeatures = []
    for i in pos:
        posWords = [feature_extraction_method(i,best_words),'pos']
        posFeatures.append(posWords)
    return posFeatures

def neg_features(neg,feature_extraction_method,best_words):
    negFeatures = []
    for j in neg:
        negWords = [feature_extraction_method(j,best_words),'neg']        
        negFeatures.append(negWords)
    return negFeatures

def clf_score(classifier,train_set,test,tag_test):
    classifier = SklearnClassifier(classifier)
    classifier.train(train_set)    
    predict = classifier.classify_many(test)
    return accuracy_score(tag_test, predict)


def cal_classifier_accuracy(train_set,test,tag_test):
    classifierlist = []
    print '各个分类器准度：'
    print 'BernoulliNB`s accuracy is %f' %clf_score(BernoulliNB(),train_set,test,tag_test)
    print 'MultinomiaNB`s accuracy is %f' %clf_score(MultinomialNB(),train_set,test,tag_test)
    print 'LogisticRegression`s accuracy is %f' %clf_score(LogisticRegression(),train_set,test,tag_test)
    print 'SVC`s accuracy is %f' %clf_score(SVC(gamma=0.001, C=100., kernel='linear'),train_set,test,tag_test)
    print 'LinearSVC`s accuracy is %f' %clf_score(LinearSVC(),train_set,test,tag_test)
    print 'NuSVC`s accuracy is %f' %clf_score(NuSVC(),train_set,test,tag_test)
    # print 'GaussianNB`s accuracy is %f' %clf_score(GaussianNB())
    classifierlist.append([BernoulliNB(),clf_score(BernoulliNB(),train_set,test,tag_test)])
    classifierlist.append([MultinomialNB(),clf_score(MultinomialNB(),train_set,test,tag_test)])
    classifierlist.append([LogisticRegression(),clf_score(LogisticRegression(),train_set,test,tag_test)])
    classifierlist.append([SVC(gamma=0.001, C=100., kernel='linear'),clf_score(SVC(gamma=0.001, C=100., kernel='linear'),train_set,test,tag_test)])
    classifierlist.append([LinearSVC(),clf_score(LinearSVC(),train_set,test,tag_test)])
    classifierlist.append([NuSVC(),clf_score(NuSVC(),train_set,test,tag_test)])
    return classifierlist
    
def find_score_max(classifier):
    max = 0
    for cla in classifier:
        if cla[1] > max:
            max = cla[1]
            object = cla[0]
    return object

def store_classifier(object,train_set,path):
    object_classifier = SklearnClassifier(object)
    object_classifier.train(train_set)
    pickle.dump(object_classifier, open(path+'/classifier.pkl','w'))
