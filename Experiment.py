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
import qns3vm
import matplotlib.pyplot as plt
import time
from seminb import *

class Experiment():

    def __init__(self, file_name, lang):

        self.__file_name = file_name
        self.__lang = lang
        self.__vectorizer = CountVectorizer(min_df = 0.0, max_df = 1.0)
        self.__vectorizer_binary = CountVectorizer(min_df = 0.0, max_df = 1.0, binary = True)
        self.__X = []
        self.__corpus = []
        self.__y = []
        self.__y_int = []
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
        annotation = []
        annotation_int = []

        for line in input_file:
            vec = line.split(';')
            annotation.append(vec[1].replace('\n',''))
            annotation_int.append(int(vec[1].replace('\n','')))
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
            self.__corpus.append(phrase.replace('\n',''))

        self.__number_of_examples = len(self.__corpus)
        transform = self.__vectorizer.fit_transform(self.__corpus)
        feature_list = self.__vectorizer.get_feature_names()

        transform_binary = self.__vectorizer_binary.fit_transform(self.__corpus)

        self.__feature_vector_len = len(feature_list)
        self.__X = self.__vectorizer.transform(self.__corpus)
        self.__X_binary = self.__vectorizer_binary.transform(self.__corpus).toarray().tolist()
        self.__y = annotation
        self.__y_int = annotation_int
        self.__set_of_classes = set(annotation)
    
    def get_status(self):
        return self.__feature_vector_len, list(self.__set_of_classes), self.__X, self.__y

    def random_subsampling(self, X_f, Xun_f, y_f,classifier): #implements random subsampling
        if(Xun_f + X_f >= 1.0):
            raise Exception("Cannot sampling if X_f + Xun_f >= 1.0")
        if(Xun_f < 0.0 or Xun_f > 1.0):
            raise Exception("Xun_f must be in the range: 0.0 <= Xun_f <= 1.0")
        if(X_f < 0.0 or X_f > 1.0):
            raise Exception("X_f must be in the range: 0.0 <= X_f <= 1.0")

        labeled_size = int(self.__number_of_examples*X_f)
        unlabeled_size = int(self.__number_of_examples*Xun_f)
        testing_size = int(self.__number_of_examples*y_f)

        all_positions = range(self.__number_of_examples)
        random.shuffle(all_positions)

        X = all_positions[0 : labeled_size]
        Xun = all_positions[labeled_size : unlabeled_size+labeled_size]
        testing = all_positions[unlabeled_size+labeled_size : testing_size+unlabeled_size+labeled_size]

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
        elif(classifier == 'S3VM'):
            for i in xrange(len(X)):
                corpus.append(self.__corpus[X[i]])
                annotation.append(self.__y_int[X[i]])
            for i in xrange(len(Xun)):
                unlabeled_corpus.append(self.__corpus[Xun[i]])
            for i in xrange(len(testing)):
                testing_corpus.append(self.__corpus[testing[i]])
                testing_annotation.append(self.__y_int[testing[i]])
            corpus = self.__vectorizer.transform(corpus)
            unlabeled_corpus = self.__vectorizer.transform(unlabeled_corpus)
            testing_corpus = self.__vectorizer.transform(testing_corpus)
        elif(classifier == 'EMNB'):
            for i in xrange(len(X)):
                corpus.append(self.__corpus[X[i]])
                if(self.__y_int[X[i]] == 1):
                    annotation.append([0,1])
                else:
                    annotation.append([1,0])
            for i in xrange(len(Xun)):
                unlabeled_corpus.append(self.__corpus[Xun[i]])
            for i in xrange(len(testing)):
                testing_corpus.append(self.__corpus[testing[i]])
                if(self.__y_int[testing[i]] == 1):
                    testing_annotation.append(1)
                else:
                    testing_annotation.append(0)
            annotation = np.array(annotation)
            testing_annotation = testing_annotation
            corpus = np.array(self.__vectorizer.transform(corpus).toarray())
            unlabeled_corpus = np.array(self.__vectorizer.transform(unlabeled_corpus).toarray())
            testing_corpus = np.array(self.__vectorizer.transform(testing_corpus).toarray())
        return corpus, annotation, unlabeled_corpus, testing_corpus, testing_annotation

    def genetic_optimization(self, number_gen, classifier, init_pop, num_survivers, iter_number, save_file = True): #genetic algorithm to optimize the params
        print '*** genetic optimation for '+classifier+' classifier ***'
        param_index = []
        result_index = []
        global_best_accuracy = 0
        global_best_index = []
        results_to_plot = []
        if(save_file):
            results_file = open('Results/result_'+classifier+str(time.time())+'.csv', 'w')
        for i in xrange(init_pop):
            param_index.append(self.get_params(classifier))
        for gen in xrange(number_gen):
            time1 = time.time()
            for i in xrange(init_pop):
                result_index.append(self.get_function_result(classifier, param_index[i], iter_number))
            result_array = np.array(result_index)
            survivers = result_array.argsort()[-num_survivers:][::-1]
            best_result = result_array[np.argmax(result_array)]
            if(global_best_accuracy < best_result):
                    global_best_accuracy = best_result
                    global_best_index = param_index[np.argmax(result_array)]
            results_to_plot.append(global_best_accuracy)
            result_index = []
            param_index = self.crossover(param_index, survivers, init_pop, classifier)
            print "Ending generation: ", gen, "in:", str(time.time() - time1)
            print "Best Result until now", global_best_accuracy
            if(save_file):
                results_file.write(str(global_best_accuracy)+';'+str(global_best_index)+'\n')
                results_file.flush()
        plt.plot(range(0, len(results_to_plot)), results_to_plot)
        plt.show()

    def get_function_result(self, classifier, index, iter_number):
        partial_result = []
        for i in xrange(iter_number):
            X, y, Xun, testing_X, testing_y = self.random_subsampling(0.1, 0.6, 0.3,classifier)
            if(classifier == 'WiSARD'):
                wisard = SemiSupervisedWiSARD(ss_confidence = index[0],
                                              ignore_zero_addr= index[1],
                                              confidence_threshold = index[2],
                                              bleaching = index[3],
                                              num_bits_addr = index[4],
                                              retina_size = self.__feature_vector_len,
                                              set_of_classes = self.__set_of_classes)
                wisard.fit(X, y, Xun)
                partial_result.append(self.SS_WiSARD_eval(wisard.predict(testing_X), testing_y))
            if(classifier == 'S3VM'):
                svm = qns3vm.QN_S3VM_Sparse(X, y, Xun, random)
                svm.parameters['lam'] = index[0]
                svm.parameters['lamU'] = index[1]
                svm.parameters['sigma'] = index[2]
                svm.parameters['kernel_type'] = index[3]
                svm.train()
                partial_result.append(self.S3VM_eval(svm.getPredictions(testing_X),testing_y))
        return np.mean(partial_result)

    def crossover(self, index, survivers, init_pop, classifier):
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
                self.mutation(classifier, aux[i]) #if it is WiSARD, edit for future classifiers
        new_index = new_index + aux
        return new_index

    def SS_WiSARD_eval(self, cls_result, testing_y):
        class_list = []
        summing = 0
        for prediction in cls_result:
            values = list(prediction.values())
            classes = list(prediction.keys())
            class_list.append(classes[values.index(max(values))])
        for i in xrange(len(class_list)):
            if(class_list[i] == testing_y[i]):
                summing += 1
        return summing/float(len(testing_y))

    def S3VM_eval(self, results, testing_y):
        summing = 0
        for i in xrange(len(results)):
            if(results[i] == testing_y[i]):
                summing += 1
        return summing/float(len(testing_y))

    def get_params(self, classifier):
        if(classifier == 'WiSARD'):
            ss_confidence = random.uniform(0.0, 1.0) #ss_confidence => (0.0,1.0)
            ignore_zero_addr = bleaching = random.choice([True, False]) #ignore_zero_addr => True or False
            confidence_threshold = random.uniform(0.0, 1.0) #confidence_threshold => (0.0,1.0)
            bleaching = random.choice([True, False]) #bleaching => True or False
            num_bits_addr = random.randint(2, 72) #num_bits_addr => discrete (2, 72)
            return ss_confidence, ignore_zero_addr, confidence_threshold, bleaching, num_bits_addr
        if(classifier == 'S3VM'):
            lam = random.random()
            lamU = random.random()
            sigma = random.random()
            kernel_type = random.choice(['Linear', 'RBF'])
            return lam, lamU, sigma, kernel_type

    def mutation(self, classifier, index):
        new_index = self.get_params(classifier)
        param = random.randint(0, len(index) - 1)
        index[param] = new_index[param]

if __name__ == "__main__":
    exp1 = Experiment("new_sts","en")
    exp1.genetic_optimization(number_gen = 2000, 
                              classifier = 'WiSARD',
                              init_pop = 200,
                              num_survivers = 50,
                              iter_number = 5)
    
    '''
    X, y, Xun, testing_X, testing_y = exp1.random_subsampling(0.1, 0.8, 0.1,'EMNB')
    tup = exp1.get_status()
    
    w1 = SemiSupervisedWiSARD(retina_size = tup[0], 
                              num_bits_addr = 5, 
                              bleaching = False, 
                              confidence_threshold = 0.5598888611857481, 
                              ignore_zero_addr = True,
                              set_of_classes = tup[1],
                              ss_confidence = 0.4801083402714187)
    w1.fit(X, y, Xun)
    print w1.get_status()
    print exp1.SS_WiSARD_eval(w1.predict(testing_X), testing_y)
    '''
    '''
    X, y, Xun, testing_X, testing_y = exp1.random_subsampling(0.1, 0.8, 0.1,'EMNB')
    tup = exp1.get_status()
    emnb = SemiNB()
    time1 = time.time()
    emnb.train_semi(np.transpose(X), y, np.transpose(Xun))
    print (time.time() - time1), 'tempo de fit'
    summing = 0
    time1 = time.time()
    for i in xrange(len(testing_X)):
        result = emnb.predict(np.transpose(testing_X[i]))
        if testing_y[i] == result:
            summing += 1
    print (time.time() - time1), 'tempo de fit'
    print summing/float(len(testing_X))
    '''