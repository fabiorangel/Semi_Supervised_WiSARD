# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from ptstemmer.implementations.OrengoStemmer import OrengoStemmer
from stemming.porter2 import stem
import re
from Preproc import Utils
import random
from SSW import SemiSupervisedWiSARD
import operator

class Experiment():

    def __init__(self, file_name, lang):

        self.__file_name = file_name
        self.__lang = lang
        self.__vectorizer = CountVectorizer(min_df = 0.0, max_df = 1.0)
        self.__vectorizer_binary = CountVectorizer(min_df = 0.0, max_df = 1.0, binary = True)
        self.__X = []
        self.__y = []
        self.__feature_vector_len = 0
        self.__set_of_classes = []
        self.__X_binary = []
        self.__number_of_examples = 0

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

        self.__number_of_examples = len(corpus)

        transform = self.__vectorizer.fit_transform(corpus)
        feature_list = self.__vectorizer.get_feature_names()

        transform_binary = self.__vectorizer_binary.fit_transform(corpus)

        self.__feature_vector_len = len(feature_list)
        self.__X = self.__vectorizer.transform(corpus)
        self.__X_binary = self.__vectorizer_binary.transform(corpus).toarray().tolist()

        self.__y = annotation
        self.__set_of_classes = set(annotation)

    def random_subsampling(self, X_f, Xun_f, classifier): #implements random subsampling
        if(Xun_f + X_f >= 1.0):
            raise Exception("Cannot sampling if X_f + Xun_f >= 1.0")
        if(Xun_f < 0.0 or Xun_f > 1.0):
            raise Exception("Xun_f must be in the range: 0.0 <= Xun_f <= 1.0")
        if(X_f < 0.0 or X_f > 1.0):
            raise Exception("X_f must be in the range: 0.0 <= X_f <= 1.0")

        labeled_size = int(self.__number_of_examples*X_f)
        unlabeled_size = int(self.__number_of_examples*Xun_f)

        all_positions = range(self.__number_of_examples)
        random.shuffle(all_positions)

        X = all_positions[0 : labeled_size]
        Xun = all_positions[labeled_size : unlabeled_size+labeled_size]
        testing = all_positions[unlabeled_size+labeled_size : ]

        corpus = []
        annotation = []
        unlabeled_corpus = []
        testing_corpus = []
        testing_annotation = []

        if(classifier == "WiSARD"): #if it is another classifier, it uses a non binary input
            for i in xrange(len(X)):
                corpus.append(self.__X_binary[X[i]])
                annotation.append(self.__y[X[i]])
            for i in xrange(len(Xun)):
                unlabeled_corpus.append(self.__X_binary[Xun[i]])
            for i in xrange(len(testing)):
                testing_corpus.append(self.__X_binary[testing[i]])
                testing_annotation.append(self.__y[testing[i]])
        else:
            for i in xrange(len(X)):
                corpus.append(self.__X[X[i]])
                annotation.append(self.__y[X[i]])
            for i in xrange(len(Xun)):
                unlabeled_corpus.append(self.__X[Xun[i]])
            for i in xrange(len(testing)):
                testing_corpus.append(self.__X[testing[i]])
                testing_annotation.append(self.__y[testing[i]])

        return corpus, annotation, unlabeled_corpus, testing_corpus, testing_annotation

    def get_best_params(self, number_gen, classifier, init_pop, num_survivers): #using genetic algorithm
        population = []
        boolean = {0: False, 1: True}
        retina_size = self.__feature_vector_len
        set_of_classes = self.__set_of_classes
        index = []
        
        if(classifier == 'WiSARD'):
            for i in xrange(init_pop):
                ss_confidence = random.uniform(0.0, 1.0) #ss_confidence => (0.0,1.0)
                ignore_zero_addr = boolean[random.randint(0,1)] #ignore_zero_addr => True or False
                confidence_threshold = random.uniform(0.0, 1.0) #confidence_threshold => (0.0,1.0)
                bleaching = boolean[random.randint(0,1)] #bleaching => True or False
                num_bits_addr = random.randint(2, 36) #num_bits_addr => discrete (2, 36)

                index.append([ss_confidence, ignore_zero_addr, confidence_threshold, bleaching, num_bits_addr])

                population.append(SemiSupervisedWiSARD(ss_confidence = ss_confidence,
                                                       ignore_zero_addr= ignore_zero_addr,
                                                       confidence_threshold = confidence_threshold,
                                                       bleaching = bleaching,
                                                       num_bits_addr = num_bits_addr,
                                                       retina_size = retina_size,
                                                       set_of_classes = set_of_classes))
                print "fitted WiSARD ", i

            for generation in xrange(number_gen):
                X, y, Xun, testing_X, testing_y = self.random_subsampling(0.7, 0.1, 'WiSARD')

                result = []
                for cls in population: #fitting WiSARDs
                    self.WiSARD_fit(cls, X, y, Xun)
                    result.append(self.SS_WiSARD_eval(cls, testing_X, testing_y))

                result = np.array(result) #do not forget to keep the best result to understand when to stop
                print "best result until now: ", result[np.argmax(result)]
                survivers = result.argsort()[-num_survivers:][::-1]
                
                if(generation < number_gen - 1): #avoiding another iteration if generations is the last
                    index = self.crossover(index, survivers, init_pop)
                    population = []
                    i = 0
                    for new_setup_parameters in index:
                        i += 1
                        ss_confidence = new_setup_parameters[0]
                        ignore_zero_addr = new_setup_parameters[1]
                        confidence_threshold = new_setup_parameters[2]
                        bleaching = new_setup_parameters[3]
                        num_bits_addr = new_setup_parameters[4]

                        population.append(SemiSupervisedWiSARD(ss_confidence = ss_confidence,
                                                               ignore_zero_addr= ignore_zero_addr,
                                                               confidence_threshold = confidence_threshold,
                                                               bleaching = bleaching,
                                                               num_bits_addr = num_bits_addr,
                                                               retina_size = retina_size,
                                                               set_of_classes = set_of_classes))
                        print "fitted WiSARD ", i
                print "Ending of Generation: ", generation
            print "best_parameter_set: ", index[np.argmax(result)], result[np.argmax(result)]
                    
        else:
            raise Exception("Classifier must be WiSARD or...")

    def crossover(self, index, survivers, init_pop):
        new_index = []
        aux = []
        number_parameters = len(index[0])

        for position in survivers:
            new_index.append(index[position])
        for i in xrange(init_pop - len(survivers)):
            p1 = random.randint(0,len(survivers) -1)
            p2 = random.randint(0,len(survivers) -1)
            new_elem = []
            for parameter_pos in xrange(number_parameters):
                if(random.random() > 0.5):
                    new_elem.append(new_index[p1][parameter_pos])
                else:
                    new_elem.append(new_index[p2][parameter_pos])
            aux.append(new_elem)
        for i in xrange(len(aux)): #mutation
            if(random.random() < 0.01):
                print 'mutation ocurried'
                self.WiSARD_mutation(aux[i])

        new_index = new_index + aux
        return new_index
    def WiSARD_mutation(self, params):
        param = random.randint(0, len(params) - 1) #find the parameter to change
        if(param == 0):
            ss_confidence = random.uniform(0.0, 1.0) #ss_confidence => (0.0,1.0)
            params[0] = ss_confidence
        elif(param == 1):
            ignore_zero_addr = boolean[random.randint(0,1)] #ignore_zero_addr => True or False
            params[1] = ignore_zero_addr
        elif(param == 2):
            confidence_threshold = random.uniform(0.0, 1.0) #confidence_threshold => (0.0,1.0)
            params[2] = confidence_threshold
        elif(param == 3):
            bleaching = boolean[random.randint(0,1)] #bleaching => True or False
            params[3] = bleaching
        elif(param == 4):
            num_bits_addr = random.randint(2, 36) #num_bits_addr => discrete (2, 36)
            params[4] = num_bits_addr
        
    def WiSARD_fit(self, classifier, X, y, Xun):
        classifier.fit(X, y, Xun)

    def SS_WiSARD_eval(self, cls, testing_X, testing_y):
        cls_result = cls.predict(testing_X) #after predicting all the testing data
        class_list = []
        summing = 0
        for prediction in cls_result:
            values = list(prediction.values())
            classes = list(prediction.keys())
            class_list.append(classes[values.index(max(values))])
        for i in xrange(len(class_list)):
            if(class_list[i] == testing_y[i]):
                summing += 1
        return summing/float(len(testing_X))

if __name__ == "__main__":

    exp1 = Experiment("hcr-train","en")
    exp1.get_best_params(number_gen = 5, 
                         classifier = 'WiSARD',
                         init_pop = 100,
                         num_survivers = 10)