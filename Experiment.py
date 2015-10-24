# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from ptstemmer.implementations.OrengoStemmer import OrengoStemmer
from stemming.porter2 import stem
import re
from Preproc import Utils
import random
from SSW import SemiSupervisedWiSARD

class Experiment():

    def __init__(self, file_name, lang):

        self.__file_name = file_name
        self.__lang = lang
        self.__vectorizer = CountVectorizer(min_df = 0.0, max_df = 1.0)
        self.__X = []
        self.__y = []
        self.__feature_vector_len = 0
        self.__set_of_classes = []

        self.__setup()

    def __setup(self):

        utl = Utils()
        stemmer = OrengoStemmer()
        stemmer.enableCaching(1000)

        input_file = open("Data_sets/"+self.__file_name+".csv", "r")

        corpus = []
        annotation = []

        for line in input_file:
            vec = line.split(';')
            annotation.append(vec[1].replace('\n',''))
            vec[0] = vec[0].lower()
            vec[0] = utl.remove_marks(vec[0])
            vec[0] = utl.replace_mentions(vec[0])
            vec[0] = utl.delete_links(vec[0])
            vec[0] = stem(vec[0])
            phrase = ''
            for elem in vec[0].split(' '):
                if(self.__lang == 'en'):
                    elem = stem(elem)
                if(self.__lang == 'pt'):
                    elem = stemmer(elem)
                phrase = phrase+' '+elem
            corpus.append(phrase.replace('\n',''))

        transform = self.__vectorizer.fit_transform(corpus)
        feature_list = self.__vectorizer.get_feature_names()

        self.__feature_vector_len = len(feature_list)
        self.__X = self.__vectorizer.transform(corpus)
        self.__y = annotation
        self.__set_of_classes = set(annotation)

    def random_subsampling():
        pass

    def get_best_params(self, number_gen, classifier, init_pop, num_survivers): #using genetic algorithm
        population = []
        boolean = {0: False, 1: True}
        retina_size = self.__feature_vector_len
        set_of_classes = self.__set_of_classes
        
        if(classifier == 'WiSARD'):
            for person in xrange(init_pop):
                ss_confidence = random.uniform(0.0, 1.0) #ss_confidence => (0.0,1.0)
                ignore_zero_addr = boolean[random.randint(0,1)] #ignore_zero_addr => True or False
                confidence_threshold = random.uniform(0.0, 1.0) #confidence_threshold => (0.0,1.0)
                bleaching = boolean[random.randint(0,1)] #bleaching => True or False
                num_bits_addr = random.randint(2, 36) #num_bits_addr => discrete (2, 36)

                population.append(SemiSupervisedWiSARD(ss_confidence = ss_confidence,
                                                       ignore_zero_addr= ignore_zero_addr,
                                                       confidence_threshold = confidence_threshold,
                                                       bleaching = bleaching,
                                                       num_bits_addr = num_bits_addr,
                                                       retina_size = retina_size,
                                                       set_of_classes = set_of_classes))
                break
            
        else:
            raise Exception("Classifier must be WiSARD or...")

    def SS_WiSARD_eval():
        pass


if __name__ == "__main__":

    exp1 = Experiment("hcr-train","en")
    exp1.get_best_params(number_gen = 10, 
                         classifier = 'WiSARD',
                         init_pop = 100,
                         num_survivers = 10)