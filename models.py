# models.py

from sentiment_data import *
from utils import *
import re
import random
import numpy as np
import heapq

from collections import Counter

def sparse_vector_dot(a: dict, b: dict):
    result = 0
    for key, value in a.items():
        if key in b:
            result += b[key] * value
    return result

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        # self.index = Indexer()
        raise Exception("Don't call me, call my subclasses")

    def extract_features_1(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        #index sentence
        # if add_to_indexer:
        #    if self.indexer.contains(word):
        #         #increase count ???
        #     else:
        #         self.indexer.add_and_get_index(word) 

        return 

        # raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=True) -> Counter:
        feature = {}
        preprocessed = self.preprocess(sentence)
        #add count and index to feature set
        for word in preprocessed:
            if self.indexer.index_of(word) != -1:
                feature[self.indexer.index_of(word)] = preprocessed[word]
        return feature

    def preprocess(self, sentence):
        preprocessed = []
        for word in sentence.words:
            preprocessed.append(re.sub(r'[^\w\s]', '', word).lower())
        while('' in preprocessed):
            preprocessed.remove('')
        return Counter(preprocessed)

    def create_vocab(self, training_ex):
        vocab = {}
        for item in training_ex:
            preprocessed = self.preprocess(item)  # item.words
        #add all to vocab
        for word in preprocessed:
            if word in vocab:
                vocab[word] += preprocessed[word]
            else:
                vocab[word] = preprocessed[word]
        # take top n results
        heap = heapq.nlargest(5000, vocab, key=vocab.get)
        # index
        for i in range(len(heap)):
            self.indexer.add_and_get_index(heap[i], add=True)
        # return


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, wieghts=[]):
        self.weights = wieghts # empty dictionary
        # self.alpa = .1
    
    def forward(self, x):
        y = sparse_vector_dot(x, self.weights)
        return 1 if y >= 0 else -1
    
    def predict(self, sentence: List[str]) -> int:
        return super().predict(sentence)

    def update(self, y_pred, y_label, x):
        # TODO move to training code
        for key, value in x.items():
            if key in self.weights:
                self.weights[key] += self.alpa * y_pred * value
            else:
                self.weights[key] = y_label - self.alpa * y_pred * value


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self):
        raise Exception("Must be implemented")


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model with updated weights
    """
    #set hyper pramemters
    epochs = 10
    #set model, indexer and extracter
    model = PerceptronClassifier()
    indexer = Indexer()
    extractor = UnigramFeatureExtractor(indexer)
    #make vocab list 
    extractor.create_vocab(train_exs)

    #enter epoch
    for epoch in epochs:
        print("the current epoch is %d" % epoch)
        accuracy = []
        #shuffle data
        shuffled_data = train_exs.shuffle()
        for item in shuffled_data:
            #extract feature
            data = item.words
            y_true = item.label
            feature =  extractor.extract_features(data)
            #classify with prceptron
            y_pred = model.predict(feature)
            #compare label and update weights
            if y_pred != y_true:
                model.update(feature)
                accuracy.append(0)
            else:
                accuracy.append(1)
        print("end of epoch %d. the acc is %f" % (epoch, np.mean(accuracy)))



def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    raise Exception("Must be implemented")


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model

index = Indexer()
feat = UnigramFeatureExtractor(index)
# print(feat.create_vocab([['The', 'Rock', 'is', 'destined', 'to', 'be', 'the', '21st', 'Century', "'s", 'new', '``', 'Conan', "''", 'and', 'that', 'he', "'s", 'going', 'to', 'make', 'a', 'splash', 'even', 'greater', 'than', 'Arnold', 'Schwarzenegger', ',', 'Jean-Claud', 'Van', 'Damme', 'or', 'Steven', 'Segal', '.']]))
