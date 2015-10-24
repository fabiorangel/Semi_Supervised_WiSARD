# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from ptstemmer.implementations.OrengoStemmer import OrengoStemmer
from stemming.porter2 import stem
import re

class Utils():

    def delete_links(self, text):
        text = re.sub('http[^\s]*', " ",text)
        text = re.sub('\s+', ' ',text)
        return text

    def replace_mentions(self, text):
        text = re.sub('@[^\s]*', " ",text)
        text = re.sub('\s+', ' ',text)
        return text

    def remove_marks(self, text):
        text = text.replace("á","a")
        text = text.replace("à","a")
        text = text.replace("ã","a")
        text = text.replace("â","a")
        text = text.replace("ç","c")
        text = text.replace("í","i")
        text = text.replace("ì","i")
        text = text.replace("î","i")
        text = text.replace("é","e")
        text = text.replace("è","e")
        text = text.replace("ê","e")
        text = text.replace("ó","o")
        text = text.replace("ò","o")
        text = text.replace("ô","o")
        text = text.replace("õ","o")
        text = text.replace("ú","u")
        text = text.replace("ù","u")
        text = text.replace("û","u")
        text = text.replace("ü","u")

        text = text.replace("Á","a")
        text = text.replace("À","a")
        text = text.replace("Ã","a")
        text = text.replace("Â","a")
        text = text.replace("Ç","c")
        text = text.replace("Í","i")
        text = text.replace("Ì","i")
        text = text.replace("Î","i")
        text = text.replace("É","e")
        text = text.replace("È","e")
        text = text.replace("Ê","e")
        text = text.replace("Ó","o")
        text = text.replace("Ò","o")
        text = text.replace("Õ","o")
        text = text.replace("Ô","o")
        text = text.replace("Ú","u")
        text = text.replace("Ù","u")
        text = text.replace("Û","u")
        text = text.replace("Ü","u")
        return text

if __name__ == "__main__":

    utl = Utils()

    input_file_name = 'hcr-train'
    lang = 'en'
    stemmer = OrengoStemmer()
    stemmer.enableCaching(1000)

    input_file = open("Data_sets/"+input_file_name+".csv", "r")

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
            if(lang == 'en'):
                elem = stem(elem)
            if(lang == 'pt'):
                elem = stemmer(elem)
            phrase = phrase+' '+elem
        corpus.append(phrase.replace('\n',''))

    vectorizer = CountVectorizer(min_df = 0.0, max_df = 1.0)
    transform = vectorizer.fit_transform(corpus)
    feature_list = vectorizer.get_feature_names()

