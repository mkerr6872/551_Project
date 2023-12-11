import numpy as np
import pickle
import nltk
import math
import os
from collections import Counter
import praw
import re


class DataPocket:
    # Holds statistics on word and sentence length as well as the source dictionary with words: counts pairs
    def __init__(self, name):
        self.wordCount = 0.0
        self.sentCount = 0.0
        self.avgWordLength = 0.0
        self.stdDevWord = 0.0
        self.avgSentLength = 0.0
        self.stdDevSent = 0.0
        self.wordFreq = Counter()
        self.source = name
        
    def __str__(self):
        return f'Source: {self.source}, avgWordLen: {self.avgWordLength}, avgSentLen: {self.avgSentLength}'
        
def logPoisson(avg, count):
    # Calculates log of a poisson pmf
    return math.log(avg**count * math.exp(-avg) / math.factorial(count))

def logGaussian(mu, sigma, x):
    # Calculates log of a gaussian pdf
    return math.log((math.exp((-1/2)*(((x-mu)/sigma)**2)) * (1/(sigma*math.sqrt(2*math.pi)))))

def dataAnalysis(document, Train):
    # Parses a text document, calculates statistics and creates the source dictionary
    # and creates and saves a Datapocket object in a .pickle file
    # document - String: text file holding data to be parsed
    # Train - Bool: flag for seperating training data and test data
    pkFile = open("objects/" + document + ".pickle", "ab")
    test = DataPocket(document)
    if Train == True:
        filePtr = open("data/" + document, "r", encoding="utf-8")
    else:
        filePtr = open("test/" + document, "r", encoding="utf-8")
    wordLength = []
    sentLength = []
    for line in filePtr:
        sentArray = nltk.sent_tokenize(line)
        for sent in sentArray:
            sent = re.sub(r"[,.;@#?!&$—–“”]+\ *", " ", sent)
            sent = re.sub(r"['’]+\ *", "", sent)
            test.sentCount +=1
            wordArray = nltk.word_tokenize(sent)
            sentLength.append(len(wordArray))
            for word in wordArray:
                if len(word) > 0:
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
    
    pickle.dump(test, pkFile)
    
def combineAnalysis():
    # combines training data from all sources to create a "standard" or "average" object
    if os.path.exists('objects/combined.pickle'):
        os.remove('objects/combined.pickle')
    test = DataPocket('combined')
    
    pkFile = open("objects/Fox.txt.pickle", 'rb')
    fox = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/CNBC.txt.pickle", 'rb')
    cnbc = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/BBC.txt.pickle", 'rb')
    bbc = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/Apple.txt.pickle", 'rb')
    apple = pickle.load(pkFile)
    pkFile.close()
    
    pkFile = open('objects/combined.pickle', 'ab')
    test.wordFreq = fox.wordFreq + cnbc.wordFreq + bbc.wordFreq + apple.wordFreq
    test.wordCount = fox.wordCount + cnbc.wordCount + bbc.wordCount + apple.wordCount
    
    pickle.dump(test, pkFile)
    
def priorMath():
    #help function to train all news sources 
    if os.path.exists('objects/BBC.txt.pickle'):
        os.remove('objects/BBC.txt.pickle')
    if os.path.exists('objects/Fox.txt.pickle'):
        os.remove('objects/Fox.txt.pickle')
    if os.path.exists('CNBC.txt.pickle'):
        os.remove('objects/CNBC.txt.pickle')
    if os.path.exists('Apple.txt.pickle'):
        os.remove('objects/Apple.txt.pickle')
        
    dataAnalysis('Fox.txt', True)
    dataAnalysis('CNBC.txt', True)
    dataAnalysis('BBC.txt', True)
    dataAnalysis('Apple.txt', True)
    
def likelihoodCalc(sample, data, tFreq, tCount):
    # Calculates the log likelifood
    #Sample - test data object
    #Data - training data object
    #tfreq - Consolidated data dictionary
    #tCount - total word count for consolidated data
    lkProb = 0
    
    for word, freq in sample.wordFreq.items():
        if word in data.wordFreq.keys():
            lkProb += logPoisson(data.wordFreq[word]*sample.wordCount/data.wordCount, sample.wordFreq[word])
        elif word in tFreq.keys():
            lkProb += logPoisson(tFreq[word]*sample.wordCount/tCount, sample.wordFreq[word])
    return lkProb
    
def sourceNewsGuess(sample):
    # Actual classifier function
    # Sample: text file with test data 
    if os.path.exists(sample+'.pickle'):
        os.remove(sample+'.pickle')
    
    dataAnalysis(sample, False)
    postProb = {}
    pkFile = open("objects/" + sample +'.pickle', 'rb')
    sampleObject = pickle.load(pkFile)
    pkFile.close()
    
    pkFile = open("objects/Fox.txt.pickle", 'rb')
    fox = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/CNBC.txt.pickle", 'rb')
    cnbc = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/BBC.txt.pickle", 'rb')
    bbc = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/Apple.txt.pickle", 'rb')
    apple = pickle.load(pkFile)
    pkFile.close()
    
    pkFile = open("objects/combined.pickle", 'rb')
    combined = pickle.load(pkFile)
    pkFile.close()
    
    totalFreq = combined.wordFreq
    totalWords = combined.wordCount

    postProb['Fox'] = likelihoodCalc(sampleObject, fox, totalFreq, totalWords)
    postProb['CNBC']= likelihoodCalc(sampleObject, cnbc, totalFreq, totalWords)
    postProb['BBC'] = likelihoodCalc(sampleObject, bbc, totalFreq, totalWords)
    postProb['Al Jazeera'] = likelihoodCalc(sampleObject, apple, totalFreq, totalWords)
    
    v = list(postProb.values())
    k = list(postProb.keys())
    
    print("Classified as :", k[v.index(max(v))])
    print("Log Likelihood:")
    print(postProb)
       
       
def verificationCheck(verFiles):
    # helper for checking validation files
    for case, tag in verFiles.items():
        print('---------------------------')
        print("Source: " + tag)
        sourceNewsGuess(case)
        os.remove("objects/" + case+'.pickle')

def commentScraper(sRed):
    # scrapes reddit comment data 
    # sRed - String: subreddit name, do not include the r/
    filept = open(sRed+".txt", "w", encoding="utf-8")
    reddit = praw.Reddit(client_id='VtNzPEQvBnXPraApATXWFg', client_secret='ZPIRrZ5XWNbgWgFvkKPhsVdfVGpTfw', user_agent='Stochastic Project')
    hot_posts = reddit.subreddit(sRed).top(limit=50)
    for post in hot_posts:
        post.comments.replace_more(limit=0)
        for top_level_comment in post.comments:
            filept.write(top_level_comment.body)
            
def redditPrep():
    # helper for scraping multiple subreddits
    commentScraper('cars')
    commentScraper('politics')
    commentScraper('army')
    commentScraper('gaming')
    
def redditAnalysis():
    # helper to create subreddit DataPockets
    if os.path.exists('objects/cars.txt.pickle'):
        os.remove('objects/cars.txt.pickle')
    if os.path.exists('objects/politics.txt.pickle'):
        os.remove('objects/politics.txt.pickle')
    if os.path.exists('objects/army.txt.pickle'):
        os.remove('objects/army.txt.pickle')
    if os.path.exists('objects/gaming.txt.pickle'):
        os.remove('objects/gaming.txt.pickle')
        
    dataAnalysis('cars.txt', True)
    dataAnalysis('politics.txt', True)
    dataAnalysis('army.txt', True)
    dataAnalysis('gaming.txt', True)
    
    combineRedditAnalysis()
    
def combineRedditAnalysis():
    #combines subreddit dictionaries and word counts
    if os.path.exists('objects/combinedReddit.pickle'):
        os.remove('objects/combinedReddit.pickle')
    test = DataPocket('combined')
    
    pkFile = open("objects/cars.txt.pickle", 'rb')
    fox = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/politics.txt.pickle", 'rb')
    cnbc = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/army.txt.pickle", 'rb')
    bbc = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/gaming.txt.pickle", 'rb')
    apple = pickle.load(pkFile)
    pkFile.close()
    
    pkFile = open('objects/combinedReddit.pickle', 'ab')
    test.wordFreq = fox.wordFreq + cnbc.wordFreq + bbc.wordFreq + apple.wordFreq
    test.wordCount = fox.wordCount + cnbc.wordCount + bbc.wordCount + apple.wordCount
    
    pickle.dump(test, pkFile)
    
def sourceRedditGuess(sample):
    #Classifier for reddit 
    #sample - text file with test data
    if os.path.exists(sample+'.pickle'):
        os.remove(sample+'.pickle')
    
    dataAnalysis(sample, False)
    postProb = {}
    pkFile = open("objects/" + sample +'.pickle', 'rb')
    sampleObject = pickle.load(pkFile)
    pkFile.close()
    
    pkFile = open("objects/cars.txt.pickle", 'rb')
    cars = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/politics.txt.pickle", 'rb')
    politics = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/army.txt.pickle", 'rb')
    army = pickle.load(pkFile)
    pkFile.close()

    pkFile = open("objects/gaming.txt.pickle", 'rb')
    gaming = pickle.load(pkFile)
    pkFile.close()
    
    pkFile = open("objects/combinedReddit.pickle", 'rb')
    combined = pickle.load(pkFile)
    pkFile.close()
    
    totalFreq = combined.wordFreq
    totalWords = combined.wordCount

    postProb['cars'] = likelihoodCalc(sampleObject, cars, totalFreq, totalWords)
    postProb['politics']= likelihoodCalc(sampleObject, politics, totalFreq, totalWords)
    postProb['army'] = likelihoodCalc(sampleObject, army, totalFreq, totalWords)
    postProb['gaming'] = likelihoodCalc(sampleObject, gaming, totalFreq, totalWords)
    
    v = list(postProb.values())
    k = list(postProb.keys())
    
    print("Classified as :", k[v.index(max(v))])
    print("Log Likelihood:")
    print(postProb)
    
def verificationRedditCheck(verFiles):
    for case, tag in verFiles.items():
        print('---------------------------')
        print("Source: " + tag)
        sourceRedditGuess(case)
        os.remove("objects/"+ case+'.pickle')

def newspaper():
    # demo to classify all newspaper tests and verification files included with code
    #only uncomment to train new model
    #priorMath()
    #combineAnalysis()
    
    verFiles = {'ver1.txt' : 'Fox', 'ver2.txt': 'CNBC', 'ver3.txt': "BBC", 'ver4.txt': 'Al Jazeera', 'ver5.txt': 'Fox'}
    testFiles = {'test1.txt': 'Fox','test2.txt': 'Fox', 'test3.txt': 'Fox', 'test4.txt': 'CNBC', 'test5.txt': 'CNBC', 'test6.txt': 'CNBC'
                 , 'test7.txt': 'BBC', 'test8.txt': 'BBC', 'test9.txt': 'BBC', 'test10.txt': 'Al Jazeera', 'test11.txt': 'Al Jazeera'
                 , 'test12.txt': 'Al Jazeera'}
    
    print('----------------Verification----------------')
    verificationCheck(verFiles)
    print('----------------Testing----------------')
    verificationCheck(testFiles)
    
def reddit():
    # demo to classify all reddit tests files included with code
    #only uncomment to train new model
    #redditPrep()
    #redditAnalysis()
    
    testFiles = {'red1.txt': 'cars','red2.txt': 'cars', 'red3.txt': 'cars', 'red4.txt': 'cars'
                 , 'red21.txt': 'cars', 'red22.txt': 'cars', 'red23.txt': 'cars', 'red24.txt': 'cars'
                 , 'red5.txt': 'politics', 'red6.txt': 'politics', 'red7.txt': 'politics', 'red8.txt': 'politics'
                 , 'red17.txt': 'politics', 'red18.txt': 'politics', 'red19.txt': 'politics', 'red20.txt': 'politics'
                 , 'red9.txt': 'army', 'red10.txt': 'army', 'red11.txt': 'army', 'red12.txt': 'army'
                 , 'red29.txt': 'army', 'red30.txt': 'army', 'red31.txt': 'army', 'red32.txt': 'army'
                 , 'red13.txt': 'gaming', 'red14.txt': 'gaming', 'red15.txt': 'gaming', 'red16.txt': 'gaming'
                 , 'red25.txt': 'gaming', 'red26.txt': 'gaming', 'red27.txt': 'gaming', 'red28.txt': 'gaming'}
    
    print('----------------Testing----------------')
    verificationRedditCheck(testFiles)
    
def displayStats(name):
    #displays stats from DataPocket object
    #name - Datapocket object
        fp = open(name, 'rb')
        data = pickle.load(fp)
        print("Source:", data.source)
        print("wordCount:",data.wordCount)
        print("sentCount:",data.sentCount)
        print("avgWordLength:",data.avgWordLength)
        print("stdDevWord:",data.stdDevWord)
        print("avgSentLength:",data.avgSentLength)
        print("stdDevSent:",data.stdDevSent)
        print('Unique words:', len(data.wordFreq))
        
def displayAll():
    # helper to display training data included with code
    buckets = ['Fox.txt.pickle', 'CNBC.txt.pickle','BBC.txt.pickle','Apple.txt.pickle'
               ,'cars.txt.pickle','politics.txt.pickle','army.txt.pickle','gaming.txt.pickle', 'combined.pickle', 'combinedReddit.pickle']
    for thing in buckets:
        displayStats(thing)

    
if __name__ == '__main__':
    newspaper()
    reddit()
    
    #displayAll()


    
    
    


    