import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
#import GaussianNB
import numpy as np
random.seed(0)
from numpy import zeros, empty, isnan, random, uint32, float32 as REAL, vstack
from gensim_models import word2vec
from gensim_models import doc2vec_modified
from gensim_models.doc2vec_modified import LabeledSentence, Doc2Vec
#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix
#nltk.download("stopwords")          # Download the stop words from nltk
import time
import os
import pickle
from gensim.models import doc2vec

# Team: Sampann Nigam & Raghava Viswanathaiah

# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
# Uncomment code block below to run from command line
#if len(sys.argv) != 3:
#  print ("python sentiment.py <path_to_data> <nlp|d2v|w2v>")
#  exit(1)
#path_to_data = sys.argv[1]
#method = sys.argv[2]

# Current Working Dir
cwd = os.getcwd()

# Comment appropriate lines depending on data and algorithm
# If running from command prompt comment all 4 lines below
#path_to_data = cwd+"/data/imdb/"
path_to_data = cwd+"/data/twitter/"
#method = "nlp"
method = "d2v"

if method == "w2v":
    path_to_pretrained_w2v = ""



def main():
    
    # Load data
    print 'Loading data..'
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

    if method == "nlp":
        print 'Running NLP..'
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)    

        
    if method == "d2v":
        print 'Running D2V..'
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)

    if method == "w2v":
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC_W2V(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC_W2V(train_pos_vec, train_neg_vec)
        
    print ("Naive Bayes")
    print ("-----------")
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)

    print ("")
    print ("Logistic Regression")
    print ("-------------------")
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)
    
# Foillwoing is performance metrics
# IMDB Data with NLP
#Naive Bayes
#-----------
#predicted:      pos     neg
#actual:
#pos             10782   1718
#neg             2075    10425
#accuracy: 84.000000
#Logistic Regression
#-------------------
#predicted:      pos     neg
#actual:
#pos             10779   1721
#neg             1998    10502
#accuracy: 85.000000

# IMDB Data with d2v 
#Naive Bayes
#-----------
#predicted:      pos     neg
#actual:
#pos             9775    2725
#neg             2722    9778
#accuracy: 78.000000
#Logistic Regression
#-------------------
#predicted:      pos     neg
#actual:
#pos             10701   1799
#neg             1820    10680
#accuracy: 85.000000

# Twitter Data with NLP
#Naive Bayes
#-----------
#predicted:      pos     neg
#actual:
#pos             64541   10459
#neg             45570   29430
#accuracy: 62.000000
#Logistic Regression
#-------------------
#predicted:      pos     neg
#actual:
#pos             64825   10175
#neg             45911   29089
#accuracy: 62.000000

# Twitter Data with D2V
#Naive Bayes
#-----------
#predicted:      pos     neg    
#actual:
#pos             53344   21641
#neg             36194   38786
#accuracy: 61.000000
#Logistic Regression
#-------------------
#predicted:      pos     neg
#actual:
#pos             55893   19092
#neg             32811   42169
#accuracy: 65.000000


def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    #path_to_dir = path_to_data #
    
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    #f= open(path_to_dir+"train-pos.txt", "r") 
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            if (len(words) == 0): continue 
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            if (len(words) == 0): continue 
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            if (len(words) == 0): continue 
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            if (len(words) == 0): continue 
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

# Function to return dictonary for count number of sentences keyword exists created 11/11 
def w_dict(corpus):
    #corpus = train_pos
    word_count_dict = {}
    for text in corpus:
        for word in text:
          if not word in word_count_dict:
            word_count_dict[word]=1
          else:
            word_count_dict[word]=word_count_dict[word]+1
    return(word_count_dict)

# Funtion to return binary  vector for given features created 11/11 
def binary_vector (corpus, features):
    X1=[] 
    for text in corpus:
        X = []
        for word in features:
          if word in text:
            X.append(1)
          else: 
            X.append(0)
        X1.append(X)  
    return(X1)

# modified on 11/11 sn
def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    word_list = []
    #word.list =  set(train_pos) | set(train_neg)
    word_list.append(train_pos)
    word_list.append(train_neg)

    # calculate the 1% count
    #positive_cnt_percent = int(<strong>len(train_pos)</strong>*0.01)
    #negative_cnt_percent = int(<strong>len(train_neg)</strong>*0.01)
    positive_cnt_percent = int(len(train_pos)*0.01)
    print  positive_cnt_percent #125
    negative_cnt_percent = int(len(train_neg)*0.01)
    print negative_cnt_percent #125
    
    # get unique words  from all words of positive and negaive training data
    unique_words = []
    unique_words = sorted(list (set(x for l in word_list[0] for x in l)|set(x for l in word_list[1] for x in l)))
    # print unique_words
    print "Unique Words:" +  str(len(unique_words)) # 73,476 unique words
    
    # Remove stop words
    filtered_words = [word for word in unique_words if word not in [w for w in stopwords]]
    print "Words Ater Removing Stopwords:" +  str(len(filtered_words))  # 73,357 words
    
     
    # Is in at least 1% of the positive texts or 1% of the negative texts
    train_pos_dict = w_dict(train_pos)
    train_neg_dict = w_dict(train_neg)
    filtered_words_1 = [word for word in unique_words if ( train_pos_dict.get(word, 0) >=  positive_cnt_percent) or ( train_neg_dict.get(word, 0) >= negative_cnt_percent) ]
    print "Words Ater Removing Words Occuring in Very Few Texts:" +  str(len(filtered_words_1))  #2,370 words
    
    # Is in at least twice as many postive texts as negative texts, or vice-versa
    filtered_words_2 = [word for word in filtered_words_1 if ( train_pos_dict.get(word, 0) >= 2* train_neg_dict.get(word, 0) ) or ( train_neg_dict.get(word, 0) >= 2* train_pos_dict.get(word, 0) )]
    print "Words Ater Removing Words Which Are Very Common In Pos and Neg sets:" +  str(len(filtered_words_2))  #498 words which will be part of feature set
    
    
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
   
    # Creating binary vectors using the featureset of words
    train_pos_vec= binary_vector(train_pos, filtered_words_2)   
    train_neg_vec= binary_vector(train_neg, filtered_words_2) 
    test_pos_vec= binary_vector(test_pos, filtered_words_2)   
    test_neg_vec= binary_vector(test_neg, filtered_words_2) 
    
    print len(train_pos_vec)
    print len(train_neg_vec)
    print len(test_pos_vec)
    print len(test_neg_vec)
    #All these vetors are length 12500 each having 0s or 1s for 498 words featureset 
    
    #Code used to test if vecors are right 
    #Y = np.array([0 for number in xrange(12500)])      
    #model1 = GaussianNB()
    #model1.fit(train_pos_vec, Y)
  
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



# Modified on 11/11
def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    labeled_train_pos = []
    labeled_train_neg = []
    labeled_test_pos = []
    labeled_test_neg = [] 
    print 'Making Labeled Sentences..'
    for i in range(0, len(train_pos)):
      labeled_train_pos.append(LabeledSentence(train_pos[i],labels=['TRAIN_POS_%s' %i]))
    for i in range(0, len(train_neg)):
      labeled_train_neg.append(LabeledSentence(train_neg[i],labels=['TRAIN_NEG_%s' %i]))
    for i in range(0, len(test_pos)):
      labeled_test_pos.append(LabeledSentence(test_pos[i],labels=['TEST_POS_%s' %i]))
    for i in range(0, len(test_neg)):
      labeled_test_neg.append(LabeledSentence(test_neg[i],labels=['TEST_NEG_%s' %i]))
    # Initialize model
    print 'Initializing Model..'
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    
    print 'Build Vocab..'
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run
    print 'Training Model..'
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)
    # As model takes lot of time to run. let us save the model to disk
    filename = 'finalized_model.sav'
    print 'Saving Model..'
    pickle.dump(model, open(filename, 'wb'))
    # Loading model from disk    
    #model = pickle.load(open(filename, 'rb'))
        
    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec  = []
    test_neg_vec  = []
    print 'Genearating training and test vectors..'
    for i in range(0, len(train_pos)):
      train_pos_vec.append(model['TRAIN_POS_%s' %i])
    for i in range(0, len(train_neg)):
      train_neg_vec.append(model['TRAIN_NEG_%s' %i])
    for i in range(0, len(test_pos)):
      test_pos_vec.append(model['TEST_POS_%s' %i])
    for i in range(0, len(test_neg)):
      test_neg_vec.append(model['TEST_NEG_%s' %i])
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec




def feature_vecs_DOC_W2V(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Load the pre-trained word2vec model
    word2vec_model = word2vec.Word2Vec.load(path_to_pretrained_w2v)

    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg

    # Use modified doc2vec codes for applying the pre-trained word2vec model
    model = doc2vec_modified.Doc2Vec(dm=0, dm_mean=1, alpha=0.025, min_alpha=0.0001, min_count=1, size=1000, hs=1, workers=4, train_words=False, train_lbls=True)
    model.reset_weights()

    # Copy wiki word2vec model into doc2vec model
    model.vocab = word2vec_model.vocab
    model.syn0 = word2vec_model.syn0
    model.syn1 = word2vec_model.syn1
    model.index2word = word2vec_model.index2word

    print "# of pre-trained vocab = " + str(len(model.vocab))



    # Extract sentence labels for the training and test data
    # YOUR CODE HERE

    sentence_labels = train_pos_labels + train_neg_labels + test_pos_labels + test_neg_labels


    new_syn0 = empty((len(sentences), model.layer1_size), dtype=REAL)
    new_syn1 = empty((len(sentences), model.layer1_size), dtype=REAL)

    syn_index = 0

    # Initialize and add a vector of syn0 (i.e. input vector) and syn1 (i.e. output vector) for a vector of a label
    for label in sentence_labels:
        v = model.append_label_into_vocab(label)  # I made this function in the doc2vec code

        random.seed(uint32(model.hashfxn(model.index2word[v.index] + str(model.seed))))

        new_syn0[syn_index] = (random.rand(model.layer1_size) - 0.5) / model.layer1_size
        new_syn1[syn_index] = zeros((1, model.layer1_size), dtype=REAL)

        syn_index += 1

    model.syn0 = vstack([model.syn0, new_syn0])
    model.syn1 = vstack([model.syn1, new_syn1])

    model.precalc_sampling()



    # Train the model
    # This may take a bit to run
    for i in range(5):
        start_time = time.time()

        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

        print "Done - Training"
        print("--- %s minutes ---" % ((time.time() - start_time) / 60))
        start_time = time.time()

        # Convert "nan" values into "0" in vectors
        indices_nan = isnan(model.syn0)
        model.syn0[indices_nan] = 0.0

        indices_nan = isnan(model.syn1)
        model.syn1[indices_nan] = 0.0

        # Extract the feature vectors for the training and test data
        train_pos_vec = [model.syn0[model.vocab["TRAIN_POS_" + str(i)].index] for i in range(len(labeled_train_pos))]
        train_neg_vec = [model.syn0[model.vocab["TRAIN_NEG_" + str(i)].index] for i in range(len(labeled_train_neg))]
        test_pos_vec = [model.syn0[model.vocab["TEST_POS_" + str(i)].index] for i in range(len(labeled_test_pos))]
        test_neg_vec = [model.syn0[model.vocab["TEST_NEG_" + str(i)].index] for i in range(len(labeled_test_neg))]

        print "Done - Extracting the feature vectors"
        print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


# modified on 11/11 sn
def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X = train_pos_vec + train_neg_vec
    
    clf = BernoulliNB()
    nb_model = clf.fit(X, Y, 1.0)
    
    logistic = LogisticRegression()
    lr_model = logistic.fit(X,Y)
    
    return nb_model, lr_model


# Modified on 11/11 
def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X = train_pos_vec + train_neg_vec
    print 'Building NB and logistics models..'
    clf = BernoulliNB()
    nb_model = clf.fit(X, Y, 1.0)
    
    logistic = LogisticRegression()
    lr_model = logistic.fit(X,Y)
    
    return nb_model, lr_model




def build_model_DOC_W2V(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    return nb_model, None

# modified on 11/11 sn
def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    #model = lr_model #
    predictions =  model.predict( test_pos_vec)
    tp = sum([1 for result in predictions if result == 'pos'  ])
    fn = len(predictions) - tp
    
    predictions =  model.predict( test_neg_vec)
    tn = sum([1 for result in predictions if result == 'neg'  ])
    fp = len(predictions) - tn
    
    accuracy = 100*(tp+tn)/(tp+tn+fp+fn)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
