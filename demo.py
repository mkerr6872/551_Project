import numpy as np
import string
import pickle
import nltk
import math
import os
from collections import Counter

class DataPocket:
    def __init__(self, name):
        self.wordCount = 0.0
        self.sentCount = 0.0
        self.avgWordLength = 0.0
        self.stdDevWord = 0.0
        self.avgSentLength = 0.0
        self.stdDevSent = 0.0
        self.wordFreq = Counter()
        self.source = name

def dataAnalysis(document):
    pkFile = open(document + ".pickle", "ab")
    test = DataPocket(document)
    filePtr = open(document, "r")
    wordLength = []
    sentLength = []
    for line in filePtr:
        sentArray = nltk.sent_tokenize(line)
        for sent in sentArray:
            sent = sent.translate(str.maketrans('', '', string.punctuation + '’“”'))
            test.sentCount +=1
            wordArray = nltk.word_tokenize(sent)
            sentLength.append(len(wordArray))
            for word in wordArray:
                if len(word) > 4:
                    test.wordCount += 1
                    wordLength.append(len(word))
                    if word[0].islower():
                        if word not in test.wordFreq:
                            test.wordFreq[word] = 1
                        else:
                            test.wordFreq[word] +=1
    
    test.avgWordLength = np.mean(wordLength)
    test.avgSentLength = np.mean(sentLength)
    test.stdDevWord = np.std(wordLength)
    test.stdDevSent = np.std(sentLength)

    #print(test.avgWordLength,test.stdDevWord, test.avgSentLength , test.stdDevSent )
    
    pickle.dump(test, pkFile)
                
def pickleTest(document):
    pkFile = open(document + ".pickle", 'rb')
    test = pickle.load(pkFile)
    print(test.source)
    print(test.avgWordLength)
    print(test.avgSentLength)
    print(test.stdDevWord)
    print(test.stdDevSent)
    print(test.wordFreq)
    
def priorMath():
    if os.path.exists('BBC.txt.pickle'):
        os.remove('BBC.txt.pickle')
    if os.path.exists('Fox.txt.pickle'):
        os.remove('Fox.txt.pickle')
    if os.path.exists('CNBC.txt.pickle'):
        os.remove('CNBC.txt.pickle')
    if os.path.exists('Apple.txt.pickle'):
        os.remove('Apple.txt.pickle')
        
    dataAnalysis('Fox.txt')
    dataAnalysis('CNBC.txt')
    dataAnalysis('BBC.txt')
    dataAnalysis('Apple.txt')
    
def likelihoodCalc(sample, data,tFreq, tCount):
    lkProb = 0
    lkProb = lkProb + logGaussian(data.avgWordLength, data.stdDevWord, sample.avgWordLength)
    lkProb = lkProb + logGaussian(data.avgSentLength, data.stdDevSent, sample.avgSentLength)
    
    for word, freq in sample.wordFreq.items():
        if word in data.wordFreq.keys():
            lkProb += logPoisson(data.wordFreq[word]*sample.wordCount/data.wordCount, sample.wordFreq[word])
        elif word in tFreq.keys():
            lkProb += logPoisson(tFreq[word]*sample.wordCount/tCount, sample.wordFreq[word])
    return lkProb
    
def sourceGuess(sample):
    if os.path.exists(sample+'.pickle'):
        os.remove(sample+'.pickle')
    
    dataAnalysis(sample)
    postProb = {}
    pkFile = open(sample +'.pickle', 'rb')
    sampleObject = pickle.load(pkFile)
    pkFile.close()
    
    pkFile = open("Fox.txt.pickle", 'rb')
    fox = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("CNBC.txt.pickle", 'rb')
    cnbc = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("BBC.txt.pickle", 'rb')
    bbc = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("Apple.txt.pickle", 'rb')
    apple = pickle.load(pkFile)
    pkFile.close()
    
    totalFreq = fox.wordFreq + cnbc.wordFreq + bbc.wordFreq + apple.wordFreq
    totalWords = fox.wordCount + cnbc.wordCount + bbc.wordCount + apple.wordCount

    postProb['Fox'] = likelihoodCalc(sampleObject, fox, totalFreq, totalWords)
    postProb['CNBC']= likelihoodCalc(sampleObject, cnbc, totalFreq, totalWords)
    postProb['BBC'] = likelihoodCalc(sampleObject, bbc, totalFreq, totalWords)
    postProb['Apple'] = likelihoodCalc(sampleObject, apple, totalFreq, totalWords)
    
    print(postProb)
    
    v = list(postProb.values())
    k = list(postProb.keys())
    
    print("Classified as :", k[v.index(max(v))])
       
def logPoisson(avg, count):
    return math.log(avg**count * math.exp(-avg) / math.factorial(count))

def logGaussian(mu, sigma, x):
    return math.log((math.exp((-1/2)*(((x-mu)/sigma)**2)) * (1/(sigma*math.sqrt(2*math.pi)))))
    
if __name__ == '__main__':
    #priorMath()
    sourceGuess('test.txt')


    