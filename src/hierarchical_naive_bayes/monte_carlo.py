"""
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import codecs
import collections
import math, random, time;
import nltk;
import numpy;
import os;
import scipy;

#from collections import defaultdict
#from nltk import FreqDist

"""
This is a python implementation of naive bayes, based on collapsed Gibbs sampling, with hyper parameter updating.
It only supports symmetric Dirichlet prior over the cluster simplex.
"""

class MonteCarlo:
    """
    """
    def __init__(self,
                 #snapshot_interval=10,
                 local_maximum_iteration=5, 
                 alpha_maximum_iteration=10,
                 hyper_parameter_sampling_interval=10):

        self._alpha_maximum_iteration = alpha_maximum_iteration
        assert(self._alpha_maximum_iteration>0)
        
        self._local_maximum_iteration = local_maximum_iteration
        assert(self._local_maximum_iteration>0)

        self._hyper_parameter_sampling_interval = hyper_parameter_sampling_interval;
        assert(self._hyper_parameter_sampling_interval>0);
        
    """
    @param num_topics: desired number of topics
    @param data: a dict data type, indexed by document id, value is a list of words in that document, not necessarily be unique
    """
    def _initialize(self, data, number_of_clusters, alpha=0.1, beta=0.1):
        self._counter=0;
        
        # define the input data
        self._data = data
        # define the number of data and the number of features
        (self._N, self._D) = self._data.shape
        # define the number of values in each feature
        self._V = numpy.zeros(self._D, dtype=numpy.int);
        
        self._K = number_of_clusters;
        
        # define the counts over different topics for all documents, first indexed by document id, the indexed by topic id
        self._data_cluster_counts = numpy.zeros((self._N, self._K));

        # define the topic assignment for every word in every language of every document, indexed by document id, language id, and word position
        self._data_feature_cluster_assignment = numpy.zeros((self._N, self._D))-1;
        
        # define the counts over words for all languages and topics, indexed by language id, topic id, and token id
        self._feature_cluster_value_counts = {};
        for feature_index in xrange(self._D):
            self._V[feature_index] = numpy.count_nonzero(numpy.unique(self._data[:, feature_index])>=0);
            self._feature_cluster_value_counts[feature_index] = numpy.zeros((self._K, self._V[feature_index]));
            print self._V[feature_index], self._feature_cluster_value_counts[feature_index].shape
        
        # initialize the vocabulary, i.e. a list of distinct tokens.
        #self._feature_type_index = feature_type_index
        #self._feature_index_type = feature_index_type
        
        #assert self._D==len(self._feature_index_type);
        #assert self._D==len(self._feature_type_index);
        
        # set the document smooth factor
        self._alpha_alpha = alpha
        # set the vocabulary smooth factor
        if beta==None or numpy.any(beta<=0):
            self._alpha_beta = 1.0/self._V;
        else:
            self._alpha_beta = beta
            
    """
    """
    def optimize_hyperparameters(self, samples=5, step=3.0):
        old_hyper_parameters = [math.log(self._alpha_alpha), math.log(self._alpha_beta)]
        
        for ii in xrange(samples):
            log_likelihood_old = self.compute_likelihood(self._alpha_alpha, self._alpha_beta)
            log_likelihood_new = math.log(random.random()) + log_likelihood_old
            #print("OLD: %f\tNEW: %f at (%f, %f)" % (log_likelihood_old, log_likelihood_new, self._alpha_alpha, self._alpha_beta))

            l = [x - random.random() * step for x in old_hyper_parameters]
            r = [x + step for x in old_hyper_parameters]

            for jj in xrange(self._alpha_maximum_iteration):
                new_hyper_parameters = [l[x] + random.random() * (r[x] - l[x]) for x in xrange(len(old_hyper_parameters))]
                trial_alpha, trial_beta = [math.exp(x) for x in new_hyper_parameters]
                lp_test = self.compute_likelihood(trial_alpha, trial_beta)

                if lp_test > log_likelihood_new:
                    #print(jj)
                    self._alpha_alpha = math.exp(new_hyper_parameters[0])
                    self._alpha_beta = math.exp(new_hyper_parameters[1])
                    #self._alpha_sum = self._alpha_alpha * self._K
                    #self._beta_sum = self._alpha_beta * self._number_of_language_types
                    old_hyper_parameters = [math.log(self._alpha_alpha), math.log(self._alpha_beta)]
                    break
                else:
                    for dd in xrange(len(new_hyper_parameters)):
                        if new_hyper_parameters[dd] < old_hyper_parameters[dd]:
                            l[dd] = new_hyper_parameters[dd]
                        else:
                            r[dd] = new_hyper_parameters[dd]
                        assert l[dd] <= old_hyper_parameters[dd]
                        assert r[dd] >= old_hyper_parameters[dd]

            print("\nNew hyperparameters (%i): %f %f" % (jj, self._alpha_alpha, self._alpha_beta))

    """
    compute the log-likelihood of the model
    """
    def compute_likelihood(self, alpha, beta):
        assert self._data_cluster_counts.shape == (self._N, self._K);
        
        alpha_sum = alpha * self._K
        beta_sum = numpy.zeros(self._D);
        for feature_index in xrange(self._D):
            beta_sum[feature_index] = beta * self._V[feature_index];

        log_likelihood = 0.0
        # compute the log likelihood of the data
        log_likelihood += scipy.special.gammaln(alpha_sum) * self._N
        log_likelihood -= scipy.special.gammaln(alpha) * self._K * self._N
        
        log_likelihood += numpy.sum(scipy.special.gammaln(alpha + self._data_cluster_counts))
        log_likelihood -= numpy.sum(scipy.special.gammaln(alpha_sum + numpy.sum(self._data_cluster_counts, axis=1)));
        
        '''
        for jj in self._data_cluster_counts.keys():
            for kk in xrange(self._K):
                log_likelihood += scipy.special.gammaln(alpha + self._data_cluster_counts[jj][kk])                    
            log_likelihood -= scipy.special.gammaln(alpha_sum + self._data_cluster_counts[jj].N())
        '''
            
        # compute the log likelihood of the feature cluster
        log_likelihood += numpy.sum(scipy.special.gammaln(beta_sum) * self._K);
        log_likelihood -= numpy.sum(scipy.special.gammaln(beta) * self._V * self._K);
        
        for feature_index in xrange(self._D):
            log_likelihood += numpy.sum(scipy.special.gammaln(beta + self._feature_cluster_value_counts[feature_index]));
            log_likelihood -= numpy.sum(scipy.special.gammaln(beta_sum[feature_index] + numpy.sum(self._feature_cluster_value_counts[feature_index], axis=1)));
            
        '''
        for feature_index in xrange(self._D):
            log_likelihood += scipy.special.gammaln(beta_sum[feature_index]) * self._K
            log_likelihood -= scipy.special.gammaln(beta) * self._number_of_language_types[feature_index] * self._K

            for jj in self._feature_cluster_value_counts[feature_index].keys():
                for kk in self._feature_type_index[feature_index]:
                    log_likelihood += scipy.special.gammaln(beta + self._feature_cluster_value_counts[feature_index][jj][kk])
                log_likelihood -= scipy.special.gammaln(beta_sum[feature_index] + self._feature_cluster_value_counts[feature_index][jj].N())
        '''

        return log_likelihood

    def sample(self):
        number_of_cluster_change = 0;
        for data_index in xrange(self._N):
            for feature_index in xrange(self._D):
                value_index = self._data[data_index, feature_index]
                
                # if the current feature value is missing, skip the sampling 
                if value_index==-1:
                    continue;
                
                #get the old cluster assignment to the feature_index in data_index at position
                old_cluster = self._data_feature_cluster_assignment[data_index, feature_index];
                if old_cluster != -1:
                    #this word_id already has a valid cluster assignment, decrease the cluster_index|data_index counts and feature_index|cluster_index counts by covering up that feature_index
                    self._data_cluster_counts[data_index, old_cluster] -= 1;
                    self._feature_cluster_value_counts[feature_index][old_cluster, value_index] -= 1;
                    
                #compute the cluster probability of current feature, given the cluster assignment for other features
                cluster_log_probability = numpy.log(self._data_cluster_counts[data_index, :] + self._alpha_alpha);
                cluster_log_probability += numpy.log(self._feature_cluster_value_counts[feature_index][:, value_index] + self._alpha_beta);
                cluster_log_probability -= numpy.log(numpy.sum(self._feature_cluster_value_counts[feature_index], axis=1) + self._V[feature_index] * self._alpha_beta);
                
                #sample a new cluster out of cluster_log_probability
                cluster_log_probability -= scipy.misc.logsumexp(cluster_log_probability);
                cluster_probability = numpy.exp(cluster_log_probability);
                temp_cluster_probability = numpy.random.multinomial(1, cluster_probability)[numpy.newaxis, :]
                new_cluster = numpy.nonzero(temp_cluster_probability==1)[1][0];
                
                if new_cluster!=old_cluster:
                    number_of_cluster_change += 1;
                
                #after sample a new cluster for that feature_index, we will change the cluster_index|data_index counts and feature_index|cluster_index counts, i.e., add the counts back
                self._data_cluster_counts[data_index, new_cluster] += 1;
                self._feature_cluster_value_counts[feature_index][new_cluster, value_index] += 1;
                
                self._data_feature_cluster_assignment[data_index, feature_index] = new_cluster
                
            if (data_index+1) % 10000==0:
                print "successfully sampled %d documents" % (data_index+1)
                
        return number_of_cluster_change

    """
    learning the corpus to train the parameters
    @param hyper_delay: defines the delay in updating they hyper parameters, i.e., start updating hyper parameter only after hyper_delay number of gibbs sampling iterations. Usually, it specifies a burn-in period.
    """
    def learning(self):
        #learning the total corpus
        #for iter1 in xrange(number_of_iterations):
        self._counter += 1;
        
        processing_time = time.time();
        
        number_of_cluster_change = self.sample();
        
        if self._counter % self._hyper_parameter_sampling_interval == 0:
            self.optimize_hyperparameters();
            
        processing_time = time.time() - processing_time;                
        print("iteration %i finished in %d seconds with %d cluster change and log-likelihood %g" % (self._counter, processing_time, number_of_cluster_change, self.compute_likelihood(self._alpha_alpha, self._alpha_beta)))

    def export_model_snapshot(self, output_directory, input_directory=None):
        exp_theta_path = os.path.join(output_directory, "exp_theta-%d.dat" % (self._counter));
        output = codecs.open(exp_theta_path, mode="w", encoding="utf-8");
        for data_id in xrange(self._N):
            for cluster_id in xrange(self._K):
                output.write("%g\t" % (self._data_cluster_counts[data_id][cluster_id]));
            output.write("\n");
        
        if input_directory==None:
            return;
        
        for feature_index in xrange(self._D):
            feature_value_index_file = os.path.join(input_directory, "feature-%d.dat" % (feature_index));
            feature_value_index_stream = codecs.open(feature_value_index_file, mode='r', encoding='utf-8');
            index_to_value = {};
            line_count = 0;
            for line in feature_value_index_stream:
                line = line.strip();
                contents = line.split("\t");
                index_to_value[line_count] = contents[0];
                line_count += 1;
            
            exp_beta_path = os.path.join(output_directory, "exp_beta-%d-feature-%d.dat" % (self._counter, feature_index));
            
            output = codecs.open(exp_beta_path, mode="w", encoding="utf-8");
            
            for cluster_index in xrange(self._K):
                output.write("==========\t%d\t==========\n" % (cluster_index));
    
                i = 0;
                #for value_index in xrange(self._V[feature_index]):
                for value_index in numpy.argsort(self._feature_cluster_value_counts[feature_index][cluster_index, :])[::-1]:
                    i += 1;
                    output.write("%s\t%g\n" % (index_to_value[value_index], (self._feature_cluster_value_counts[feature_index][cluster_index, value_index]+self._alpha_beta)/(numpy.sum(self._feature_cluster_value_counts[feature_index][cluster_index, :])+self._alpha_beta*self._V[feature_index])));
                
            output.close();
        
    """
    this methods change the count of a topic in one doc_id and a word of one topic by delta
    this values will be used in the computation
    @param doc_id: the doc_id id
    @param word: the word id
    @param topic: the topic id
    @param delta: the change in the value
    @deprecated:
    """
    '''
    def change_count(self, doc_id, lang_id, word_id, topic_id, delta):
        self._data_cluster_counts[doc_id].inc(topic_id, delta)
        self._feature_cluster_value_counts[lang_id][topic_id].inc(word_id, delta)
    '''
        
if __name__ == "__main__":
    print "not implemented"