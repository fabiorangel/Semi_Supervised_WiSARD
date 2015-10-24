import numpy as np
from matplotlib import pyplot as plt
from PyWANN.WiSARD import *
import time
import multiprocessing

class SemiSupervisedWiSARD(): # Input: labeled examples and unlabeled examples - Output: classifier
    def __init__ (self, 
                  retina_size, 
                  num_bits_addr, 
                  bleaching, 
                  confidence_threshold, 
                  ignore_zero_addr,
                  set_of_classes,
                  ss_confidence):

        self.__retina_size = retina_size
        self.__num_bits_addr = num_bits_addr
        self.__is_bleaching = bleaching
        self.__confidence_threshold = confidence_threshold
        self.__is_ignoring_zero_addr = ignore_zero_addr
        self.__set_of_classes = set_of_classes
        self.__ss_confidence = ss_confidence

        self.__main_wisard =  WiSARD(retina_size = self.__retina_size,
                                     num_bits_addr = self.__num_bits_addr,
                                     bleaching = self.__is_bleaching,
                                     confidence_threshold = self.__confidence_threshold,
                                     ignore_zero_addr = self.__is_ignoring_zero_addr)
        self.__setup()
        

    def fit(self, X = [], y = [], unlabeled_X = []):

        if(type(X) is not list):
            raise Exception("Type of X must be a list of examples")
        if(type(unlabeled_X) is not list):
            raise Exception("Type of unlabeled_X must be a list of examples")
        if(len(X) != len(y)):
            raise Exception("Number of Examples must match number of labels")
        if(len(X[0]) != self.__retina_size):
            raise Exception("Size of example must have the same size as retina:\n"+
                                "Size of example = %d, Size of Retina = %d"%(len(unlabeled_X[0]), self.__retina_size))
        if(unlabeled_X != []):
            if(len(unlabeled_X[0]) != self.__retina_size):
                raise Exception("Size of example must have the same size as retina:\n"+
                                "Size of example = %d, Size of Retina = %d"%(len(unlabeled_X[0]), self.__retina_size))
        #here the fit begins
        if(X != []):
            for position in xrange(len(X)):
                class_name = y[position]
                example = X[position]
                self.__main_wisard.add_training(class_name, example)

        if(unlabeled_X != []):
            for position in xrange(len(unlabeled_X)): #here comes the unsupervised part. Learning with unlabeled data.
                possible_classes = self.__main_wisard.classify(unlabeled_X[position]) #result of the classifying #returns a dictionary
                values = possible_classes.values() #list of results
                index = possible_classes.keys() #list of classes
                list_possible_classes = []

                for i in xrange(len(values)): #pairing results and classes
                    list_possible_classes.append([index[i], values[i]])

                list_possible_classes = sorted(list_possible_classes, key=lambda a_entry: a_entry[1]) #sorting list by the values
                if(list_possible_classes[-1][1] != 0): #best result must be diferent from zero
                    ss_confidence = 1 - float(list_possible_classes[-2][1])/float(list_possible_classes[-1][1]) #calculating confidence
                else:
                    ss_confidence = 0.0 #if best result == 0, confidence is zero
                if(ss_confidence >= self.__ss_confidence): #if I have confidence in the result, i'll train this example to the selected class
                    class_name = list_possible_classes[-1][0]
                    self.__main_wisard.add_training(class_name, unlabeled_X[position])         

    def __setup(self):
        for class_name in self.__set_of_classes:
            self.__main_wisard.create_discriminator(class_name)

    def predict(self, testing_corpus):
        result = []
        for position in xrange(len(testing_corpus)):
            result.append(self.__main_wisard.classify(testing_corpus[position]))
        return result


if __name__ == "__main__":
    
    teste = SemiSupervisedWiSARD(retina_size = 3,
                                 num_bits_addr = 2,
                                 bleaching = True,
                                 confidence_threshold = 0.5,
                                 ignore_zero_addr = True,
                                 set_of_classes = ["1","2"],
                                 ss_confidence = 0.5)
    
    teste.fit([[1,0,0],[0,0,0],[0,1,1]], ["1", "2", "1"], [[0,1,0]])
    teste.fit([[1,1,0]], ["1"])

    list_possible_classes = [["1", 3], ["2", 4], ["3", 2]]
    print list_possible_classes
    list_possible_classes = sorted(list_possible_classes, key=lambda a_entry: a_entry[1]) #sorting list by the values
    print list_possible_classes