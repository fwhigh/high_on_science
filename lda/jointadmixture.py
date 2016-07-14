#!/usr/bin/env/python 

import numpy as np
from numpy import logical_not, logical_and, logical_or, log, exp, array, ones, zeros
import scipy
from itertools import izip, izip_longest
import sys
import os
#from sklearn.metrics import roc_curve, auc
from oscardata import OscarData
import cPickle as pickle
import datetime
#sys.path.append(os.path.abspath("../ldasampler/"))
import text
import gzip

NTOPICS = 100
TRAIN_SIZE = 10000
TEST_SIZE = 1000
#TRAIN_SIZE = 100
#TEST_SIZE = 100


def l2regress(features, response, lamb=1.0):
    n_feats = features.shape[1]
    features = features[np.logical_not(np.isnan(response)),:]
    response = response[np.logical_not(np.isnan(response))]
    coefs, resid, rank, sing  = np.linalg.lstsq(features.T.dot(features) + lamb * np.identity(n_feats), features.T.dot(response))
    # clip the coefficients -- don't do this any more now we have an intercept
    #coefs = np.clip(coefs, -100, 0)
    # work out the average prediction
    #avg = features.dot(coefs).mean()
    # assume the intercept is the last column and adjust
    #coefs[-1] -= (avg - response.mean())
    #print "Mean AVS %.2f\tresid %.2f" % (avg, resid.mean())
    # return for compatible signature with lstsq()
    return coefs, resid, rank, sing


def match(keys, into):
    """Takes two vectors. Returns the index into the second where each element of the first is found."""
    assert(np.all(np.in1d(keys, into)))
    sort = into.argsort()
    indices = np.searchsorted(into, keys, sorter=sort)
    indices = sort[indices]
    for i,ix in enumerate(indices):
        assert(into[ix] == keys[i])
    return indices


def read_data(wiki='/volmount/tagpredict/wikipedia_plots_20160606.dat.gz.npz',
    oscar='/volmount/tagpredict/oscardump_newest.tsv', TRAIN_SIZE=TRAIN_SIZE, TEST_SIZE=TEST_SIZE):
    (wiki_ids, wiki_data, wiki_vocab, collection_data, collections) = text.load_data(wiki)
    oscar_data = OscarData(oscar)#, trial_subset=True)

    ids = np.intersect1d(wiki_ids, oscar_data.movie_ids)
    ids = ids[np.random.choice(ids.shape[0], TRAIN_SIZE+TEST_SIZE, replace=False)]

    np.random.shuffle(ids)

    train_ids = ids[:TRAIN_SIZE]
    test_ids = ids[TRAIN_SIZE:(TRAIN_SIZE+TEST_SIZE)]

    wiki_train = wiki_data[match(train_ids, wiki_ids),:]
    oscar_train = oscar_data.copy()
    oscar_train.select_rows(match(train_ids, oscar_train.movie_ids))  # select the indices
    oscar_train.select_columns(oscar_train.predictors)

    wiki_test = wiki_data[match(test_ids, wiki_ids),:]
    oscar_test = oscar_data.copy()
    oscar_test.select_rows(match(test_ids, oscar_test.movie_ids))  # select the indices
    oscar_test.select_columns(oscar_test.predictors)

    if oscar_data.data.shape[0] < (TRAIN_SIZE+TEST_SIZE):
        TRAIN_SIZE = int(oscar_data.data.shape[0] * (float(TRAIN_SIZE) / (TRAIN_SIZE+TEST_SIZE)))
        TEST_SIZE = oscar_data.data.shape[0] - TRAIN_SIZE

    observed_data = np.clip(oscar_train.data, a_min=0, a_max=1)

    assert(np.all(oscar_train.data[oscar_train.mask] >= 0) and np.all(oscar_train.data[oscar_train.mask] <= 1))
    assert(np.all(oscar_test.data[oscar_test.mask] >= 0) and np.all(oscar_test.data[oscar_test.mask] <= 1))

    print "Training on {} data points".format(wiki_train.shape[0])
    print "Testing on {} data points".format(wiki_test.shape[0])

    return oscar_train, wiki_train, oscar_train, oscar_test, wiki_vocab


def read_avs(titles, avsfile='/volmount/tagpredict/avs_20160702.csv.gz'):
    """read in AVS or another predictor for an array of title IDs"""
    avs = np.zeros(len(titles))
    #print indices
    indices = {title:i for i,title in enumerate(titles)}
    with gzip.open(avsfile) as infile:
        for line in infile:
            title,score = line.strip().split(',')
            title = int(title)
            #print title
            try:
                # let's exclude titles with AVS of 0
                score = float(score)
                if score != 0.0 and title in indices:
                    avs[indices[title]] = score
            except ValueError, e:
                print e
                pass
    return avs


def new_predictions(coefs, topic_counts):
    """give a k-vector of the predictions for this doc if 1 was added to each of k topics"""
    tmp_mat = topic_counts + np.identity(topic_counts.shape[0])  # create a matrix adding one to a different count in each row
    tmp_mat = (tmp_mat / (tmp_mat.sum(1)).astype(float))  # convert to a probability dist
    tmp_mat = np.concatenate((tmp_mat, np.ones(tmp_mat.shape[0])[:,np.newaxis]), 1)  # add the intercept
    return tmp_mat.dot(coefs)


def doc_topic_means(Cdt):
    """return a Nd * ntopics+1 matrix of count(topic)/doc_len for each doc, plus an intercept column at the end"""
    ret = np.ones((Cdt.shape[0], Cdt.shape[1]+1))
    for d in xrange(Cdt.shape[0]):
        ret[d,:-1] = Cdt[d,:] / Cdt[d,:].sum()
    return ret


class Sampler(object):
    def __init__(self):
        self.alpha = .5  # prior for doc-topic dist (should maybe be function of ntopics)?
        self.beta = [.5, .5]  # prior for each feature in Oscar
        self.gamma = .01  # prior for word dist in text topics
        self.avs = None
        self.predictors = None


    def load_data(self, oscar_data, wiki_data, wiki_vocab):
        self.wiki_vocab = wiki_vocab
        self.Nd = oscar_data.data.shape[0]  # number of movies
        self.Nx = oscar_data.data.shape[1]  # number of oscar features
        self.Nw = wiki_data.shape[1]  # number of word features
        self.doc_len = np.array(wiki_data.sum(axis=1)).ravel()  # number of words in each document
        self.data_mask = oscar_data.data >= 0
        self.oscar_data = oscar_data

        (d, w) = wiki_data.nonzero()
        assert(np.array_equal(d, np.sort(d)))  # check the order of documents is correct
        self.words = np.repeat(w, repeats=np.array(wiki_data[d,w]).ravel())  # flattened version of wiki_data


    def use_avs(self, avs_scores):
        self.avs = np.log(avs_scores+.00001)
        self.avs[avs_scores==0] = np.nan


    def use_predictor(self, predictor):
        self.predictors = predictor


    def count_topics(self):
        # tally up the counts -- use floats because we will add the priors directly; only count observed features
        tag_topics = self.tag_topics
        word_topics = self.word_topics
        words = self.words
        doc_len = self.doc_len
        ntopics = self.ntopics
        # Cxft is the matrix of counts of each bernoulli outcome in each feature in each topic
        Cxft = np.zeros((2, self.Nx, ntopics), dtype=float)
        for x in xrange(2):
            for f in xrange(self.Nx):
                Cxft[x,f,:] = np.bincount(tag_topics[:,f][logical_and(self.data_mask[:,f], self.oscar_data.data[:,f]==x)],
                                          minlength=ntopics) + self.beta[x]
        # Cwt is the matrix of counts of each word in each topic
        Cwt = np.zeros((self.Nw, ntopics), dtype=float)
        for t in xrange(ntopics):
            Cwt[:,t] = np.bincount(words[word_topics==t],
                                   minlength=self.Nw) + self.gamma
        # Cdt is the matrix of counts of each topic in each movie
        Cdt = np.zeros((self.Nd, ntopics), dtype=float)
        d_i = 0
        for d in xrange(self.Nd):
            # the counts in the tag data -- add this back in after testing
            Cdt[d,:] = np.bincount(tag_topics[d,self.data_mask[d,:]],
                                   minlength=ntopics)
            # the counts in the word data
            Cdt[d,:] += np.bincount(word_topics[d_i:(d_i+doc_len[d])],
                                    minlength=ntopics) + self.alpha
            d_i += doc_len[d]
        return Cdt, Cxft, Cwt


    def initialize(self, ntopics):
        self.iteration = 0
        self.ntopics = ntopics
        # the topic assignments (start from random state)
        # tag_topics is a matrix like the input data (each possible movie-feature combination)
        self.tag_topics = np.random.randint(ntopics, size=(self.Nd, self.Nx))
        # word topics is a multinomial outcome, so each document has a different length
        # use a single vector and index into it using doc_len
        self.word_topics = np.random.randint(ntopics, size=self.doc_len.sum())
        print ntopics, "topics"


    def fit(self, iterations=300, ignore_wiki=False):
        Cdt, Cxft, Cwt = self.count_topics()
        i = 0
        for it in xrange(0, iterations):
            print "(joint admixture) Iteration", self.iteration
            mean_z = doc_topic_means(Cdt) # TODO -- testing adding the intercept back after regularizing [:,:-1]
            #newpreds = np.concatenate((mean_z, oscar_data.data[:,extra_preds]), 1)

            # regress the AVS score on the empirical topic weights
            #coefs, resid, rank, sing = np.linalg.lstsq(mean_z, resp)
            if self.avs is not None:  
                print "AVS prediction"
                if self.predictors is not None: # TODO: this is probably broken after adding the intercept back in, test
                    predictors = np.concatenate((mean_z, self.predictors[:,np.newaxis]), 1)
                else:
                    predictors = mean_z
                #coefs, resid, rank, sing = np.linalg.lstsq(predictors, self.avs)
                coefs, resid, rank, sing = l2regress(predictors, self.avs, lamb=4.0)
                #coefs = coefs[:-extra_preds.sum()]
                if self.predictors is not None:
                    extra_coef = coefs[-1]
                    coefs = coefs[:-1]
                else:
                    extra_coef = 0
                self.coefs = coefs    

            w_i = 0  # this points to the position in words or word_topics
            # iterate over all text documents
            for d in xrange(self.Nd):
                if d % 1000 == 0:
                    print "\tmovie %d (words)" % d
                if self.avs is not None and not np.isnan(self.avs[d]):
                    correction = coefs / (Cdt[d,:].sum() - 1)  # AVS correction
                    correction = correction[:-1]  # remove intercept
                for _ in xrange(self.doc_len[d]):
                    w = self.words[w_i]
                    z = self.word_topics[w_i]
                    Cwt[w,z] -= 1
                    Cdt[d,z] -= 1
                    
                    # unnormalized: P(observed word|topic) * count of topic in doc
                    topic_probs = (Cwt[w,:] / Cwt.sum(axis=0)) * Cdt[d,:]
                    # add in AVS
                    if self.avs is not None and not np.isnan(self.avs[d]):
                        new_preds = new_predictions(coefs, Cdt[d,:])
                        if self.predictors is not None:
                            new_preds += extra_coef * self.predictors[d]
                        topic_probs *= exp(2 * correction
                                       * (self.avs[d] - new_preds)
                                       - correction**2)
                    # sample by generating a number from 0 to the sum of the unnormalized dist
                    topic_probs = topic_probs.cumsum()
                    # no need to sort because cumsum produces sorted output; also, the last element is the max:
                    z = np.searchsorted(topic_probs,
                                        np.random.random() * topic_probs[-1])
                    Cwt[w,z] += 1
                    Cdt[d,z] += 1
                    self.word_topics[w_i] = z
                    w_i += 1
            # iterate over all observed values in Oscar
            lastprinted = -1
            for (d, feat) in izip(*self.data_mask.nonzero()):
                if lastprinted != d and d % 1000 == 0:
                    print "\tmovie %d (tags)" % d
                    lastprinted = d 
                i += 1
                x = self.oscar_data.data[d, feat]
                z = self.tag_topics[d, feat]
                #assert(Cxft[x,feat,z]>=1)
                #assert(Cdt[d,z]>=1)

                Cxft[x,feat,z] -= 1
                Cdt[d,z] -= 1

                # unnormalized: P(observed x|feat, topic) * count of topic in doc
                topic_probs = (Cxft[x,feat,:] / Cxft[:,feat,:].sum(axis=0)) * Cdt[d,:]
                if self.avs is not None and not np.isnan(self.avs[d]):
                    # multiply in the factor due to error of the regression prediction
                    new_preds = new_predictions(coefs, Cdt[d,:])
                    if self.predictors is not None:
                        new_preds += extra_coef * self.predictors[d]
                    topic_probs *= exp(2 * correction
                                       * (self.avs[d] - new_predictions(coefs, Cdt[d,:]))
                                       - correction**2)
                # sample by generating a number from 0 to the sum of the unnormalized dist
                topic_probs = topic_probs.cumsum()
                # no need to sort because cumsum produces sorted output; also, the last element is the max:
                z = np.searchsorted(topic_probs,
                                    np.random.random() * topic_probs[-1])

                Cxft[x,feat,z] += 1
                Cdt[d,z] += 1
                #assert(z<ntopics)
                self.tag_topics[d,feat] = z
            self.iteration += 1




    def fold_in(self, oscar_data, wiki_data):
        results = np.zeros((oscar_data.shape[0], self.ntopics))
        #self.tag_topics = np.random.randint(ntopics, size=(self.Nd, self.Nx))
        # word topics is a multinomial outcome, so each document has a different length
        # use a single vector and index into it using doc_len
        #self.word_topics = np.random.randint(ntopics, size=self.doc_len.sum())
        ntopics = self.ntopics
        nfeats = self.Nx
        nwords = self.Nw
        Nd_foldin = wiki_data.shape[0]
        (d, w) = wiki_data.nonzero()
        assert(np.array_equal(d, np.sort(d)))  # check the order of documents is correct
        words_foldin = np.repeat(w, repeats=np.array(wiki_data[d,w]).ravel())  # flattened version of wiki_data
        doc_len_foldin = np.array(wiki_data.sum(axis=1)).ravel()
        tag_topics_foldin = np.random.randint(ntopics, size=(Nd_foldin, self.Nx))
        word_topics_foldin = np.random.randint(ntopics, size=doc_len_foldin.sum())
        data_mask = oscar_data != -1

        Cdt, Cxft, Cwt = self.count_topics()
        max_it = 50
        
        for d in xrange(oscar_data.shape[0]):
            print d
            tag_topics = tag_topics_foldin[d,:]  # topic assignments for fold in feats
            word_topics = word_topics_foldin[doc_len_foldin[d]:]
            # equivalent to Cdt[d,:] in fit()
            topic_counts = (np.bincount(tag_topics_foldin[d,data_mask[d,:]].ravel(), minlength=ntopics)
                + np.bincount(word_topics_foldin, minlength=ntopics))
            #word_topics = np.random.randint(ntopics, size=wiki_data[d,:].sum())  # topic assignments for fold in text
            # add in counts from this doc just while we are folding it in
            #Cwt += np.bincount(word_topics, minlength=ntopics)
            #Cxft += np.bincount(tag_topics, minlength=ntopics) 
            
            for it in xrange(max_it):
                # iterate over the words in the document
                w_i = doc_len_foldin[:d].sum()
                while w_i < doc_len_foldin[d]:
                    w = words_foldin[w_i]
                    z = word_topics_foldin[w_i]
                    topic_counts[z] -= 1
                
                    # unnormalized: P(observed word|topic) * count of topic in doc
                    topic_probs = (Cwt[w,:] / Cwt.sum(axis=0)) * topic_counts
                    # sample by generating a number from 0 to the sum of the unnormalized dist
                    topic_probs = topic_probs.cumsum()
                    # no need to sort because cumsum produces sorted output; also, the last element is the max:
                    z = np.searchsorted(topic_probs,
                                        np.random.random() * topic_probs[-1])
                    topic_counts[z] += 1
                    word_topics_foldin[w_i] = z

                    w_i += 1
                
                # iterate over all features
                for feat in xrange(nfeats):
                    if not data_mask[d, feat]:  # skip unobserved data
                        continue
                    x = oscar_data[d, feat]
                    z = tag_topics_foldin[d, feat]
                    topic_counts[z] -= 1
                    #assert(Cxft[x,feat,z]>=1)
                    #assert(Cdt[movie,z]>=1)

                    # unnormalized: P(observed x|feat, topic) * count of topic in doc
                    topic_probs = (Cxft[x,feat,:] / Cxft[:,feat,:].sum(axis=0)) * topic_counts
                    # sample by generating a number from 0 to the sum of the unnormalized dist
                    topic_probs = topic_probs.cumsum()
                    # no need to sort because cumsum produces sorted output; also, the last element is the max:
                    z = np.searchsorted(topic_probs,
                                        np.random.random() * topic_probs[-1])

                    #assert(z<ntopics)
                    tag_topics_foldin[d,feat] = z
                    topic_counts[z] += 1
                    
            results[d,:] = (np.bincount(tag_topics_foldin[d,data_mask[d,:]].ravel(), minlength=ntopics)
                + np.bincount(word_topics_foldin, minlength=ntopics))
        return results

                # todo: deal with topic assignments for unobserved features?


    #def log_likelihood(self):
    #    return self.log_likelihood_heldout(self.tag_topics, self.word_topics, self.doc_len, self.oscar_data, self.wiki_data)


    def log_likelihood_heldout(self, tag_topics, word_topics, doc_len, oscar_data, wiki_data):
        Cdt, Cxft, Cwt = self.count_topics()
        Pft = (Cxft / Cxft.sum(axis=0))[1,:,:]
        Pwt = Cwt / Cwt.sum(axis=0)
        data_mask = oscar_data.data >= 0
        ntopics = self.ntopics
        betafn = scipy.special.beta
        (d, w) = wiki_data.nonzero()
        assert(np.array_equal(d, np.sort(d)))  # check the order of documents is correct
        words = np.repeat(w, repeats=np.array(wiki_data[d,w]).ravel())  # flattened version of wiki_data
        doc_len = np.array(wiki_data.sum(axis=1)).ravel()
        ll = .0
        w_i = 0
        for d in xrange(tag_topics.shape[0]):
            doc_tag_topics = tag_topics[d,:]  # topic assignments for fold in feats
            doc_word_topics = word_topics[w_i:w_i+doc_len[d]]
            topic_counts = (np.bincount(tag_topics.ravel()[data_mask[d,:]], minlength=ntopics)
                + np.bincount(word_topics, minlength=ntopics))
            # estimate P of doc topic weights using dirichlet-multinomial distribution
            nonzerocounts = topic_counts[topic_counts>0]
            n = nonzerocounts.shape[0]
            ll += np.log(betafn(self.alpha * ntopics, n) / (nonzerocounts * betafn(self.alpha, n)).prod())
            # now add the LL for the tags
            on_feats = doc_tag_topics.copy()
            on_feats[np.logical_or(oscar_data.data[d,:]==0, np.logical_not(data_mask[d,:]))] = -1
            off_feats = doc_tag_topics.copy()
            off_feats[np.logical_or(oscar_data.data[d,:]==1, np.logical_not(data_mask[d,:]))] = -1
            for topic in xrange(ntopics):
                ll += np.log(Pft[:,topic][on_feats==topic]).sum()
                ll += np.log((1-Pft[:,topic])[off_feats==topic]).sum()
            # now for the words
            text = words[w_i:w_i+doc_len[d]]
            topics = word_topics[w_i:w_i+doc_len[d]]
            for topic in xrange(ntopics):
                ll += np.log(Pwt[:,topic][text[topics==topic]]).sum()
            w_i += doc_len[d]
        return ll / oscar_data.data.shape[0]

    def save(self, filename):
        with gzip.open(filename, 'w') as outfile:
            pickle.dump(self, outfile)


    def load_fit(self, filename):
        with gzip.open(filename) as infile:
            other = pickle.load(infile)
        self.ntopics = other.ntopics
        self.tag_topics = other.tag_topics
        self.word_topics = other.word_topics
        self.avs = other.avs
        self.predictors = other.predictors
        if 'coefs' in dir(other):
            self.coefs = other.coefs
        self.alpha = other.alpha
        self.beta = other.beta
        self.gamma = other.gamma
        self.oscar_data = other.oscar_data
        self.wiki_vocab = other.wiki_vocab
        self.Nd = other.Nd
        self.Nx = other.Nx
        self.Nw = other.Nw
        self.doc_len = other.doc_len
        self.data_mask = other.data_mask
        self.words = other.words
        self.iteration = other.iteration

                   

if __name__=='__main__':
    if len(sys.argv) < 3:
        sampler = Sampler()
        oscar_train, wiki_train, oscar_test, wiki_test, wiki_vocab = read_data()
        sampler.load_data(oscar_train, wiki_train, wiki_vocab)
        sampler.initialize(NTOPICS)
        with open(sys.argv[1] + 'data.pkl.gz', 'w') as outfile:
            pickle.dump([oscar_train, wiki_train, oscar_test, wiki_test, wiki_vocab], outfile)
    else:
        sampler = Sampler(sys.argv[1])
        with open(sys.argv[1]) as infile:
            oscar_train, wiki_train, oscar_test, wiki_test, wiki_vocab  = pickle.load(infile)

    for i in xrange(30):
        sampler.fit(100)
        sampler.save(sys.argv[1] + 'model.pkl.gz')
        print sampler.log_likelihood_heldout(sampler.tag_topics, sampler.word_topics, sampler.doc_len, oscar_train, wiki_train)#oscar_test, wiki_test)
