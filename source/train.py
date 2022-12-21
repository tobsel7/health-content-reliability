import os
import re
import csv
import sys
import time
import pickle
import argparse
import svmlight
import numpy as np
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter
from sklearn import preprocessing
from nltk.corpus import stopwords
from w3lib.html import remove_tags
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.datasets import dump_svmlight_file, load_svmlight_file

'''
This function loads the privacy links lexicon
'''
def load_privacy_words():
        with open("lexicon/privacy.txt") as file:
                privacy = [word.rstrip() for word in file.readlines()]

        return privacy

'''
This function loads the contact links lexicon
'''
def load_contact_words():
        with open("lexicon/contact.txt") as file:
                contact = [word.rstrip() for word in file.readlines()]

        return contact


'''
This function loads the stopwords lexicon
'''
def load_stopwords():
        with open("lexicon/stopwords.txt") as file:
                stopwords = [word.rstrip() for word in file.readlines()]

        return stopwords

'''
This function loads the commercial words lexicon
'''
def load_commercial_words():
        with open("lexicon/comm_list.txt") as file:
                comm_list = [word.rstrip() for word in file.readlines()]

        return comm_list

# static variables
STOPWORDS = set(stopwords.words("english"))
NEW_WORDS = load_stopwords()
STOPWORDS = STOPWORDS.union(NEW_WORDS)
COMMERCIAL = load_commercial_words()
CONTACT = load_contact_words()
PRIVACY = load_privacy_words()


'''
This function saves in a file the obtained performance for a concrete cost-factor and feature set
'''
def save_results(dataset, features, cost_factor, ts, accuracies, f1_l, f1_rel_l, f1_unrel_l):
        if not os.path.exists('./results'):
                os.makedirs('./results')
        
        with open("./results/"+dataset+"_results_"+features+"_cost_fact"+str(cost_factor+1)+"_"+ts+".txt", "w+") as f:
                f.write("The mean accuracy is "+str(np.mean(accuracies))+"\n")
                f.write("The f1-score is "+str(np.mean(f1_l))+"\n")
                f.write("The credible f1-score is "+str(np.mean(f1_rel_l))+"\n")
                f.write("The non-credible f1-score is "+str(np.mean(f1_unrel_l))+"\n")

'''
This function evaluates the obtained predictions and returns the number of true positives, true negatives, false positives and false negatives
'''
def evaluate(predictions):
        tp, tn, fp, fn = 0, 0, 0, 0
        for a, b in zip(val,predictions):
                if np.sign(a) == np.sign(b): # true
                        if np.sign(a) == -1:
                                tn +=1
                        else:
                                tp += 1
                else: # false
                        if np.sign(a) == 1:
                                fn += 1
                        else: 
                                fp += 1

        return tp, tn, fp, fn


'''
This function removes the label from the test instances and replaces it with 0 values (svmlight specific format)
'''
def adapt_to_svmlight_format(aux):
        test = []
        val = []
        
        for element in aux:
                lst = list(element)
                val.append(lst[0])
                lst[0] = 0
                element = tuple(lst)
                test.append(element)

        return test, val

'''
This function implements the weighted accuracy metric described in Sondhi's study
'''
def weighted_accuracy(bias, tn, tp, fn, fp):
        return (bias*tp+tn)/(bias*(tp+fn)+tn+fp) 

'''
This function loads a svmlight file and parses it into its specific format
'''
def svm_parse(filename):
    
    features,target = load_svmlight_file(filename)
    _,nfeatures = features.shape
    
    it = 0
    for cl in target:
            doc_features = []
            for i in range(nfeatures):
                doc_features.append((float(i+1),features[it,i]))        
            it+=1
            yield (int(cl),doc_features)

'''
This function calculates the word-based features as their normalized frequency value
'''
def word_features(doc, vectorizer):
        vector = vectorizer.transform([doc])
        doc_to_list = list(vector.toarray()[0])
        maximum = max(doc_to_list)

        if maximum:
                for val in doc_to_list:
                        index = doc_to_list.index(val)
                        doc_to_list[index] = val/maximum
        
        return doc_to_list

'''
This function counts the total commercial interest words appearances and returns the normalized frequency total value
'''
def count_commercial_keywords(filename, doc):
        commercial_words = 0
        
        with open(filename,encoding="utf-8",errors="ignore") as reader:
                soup = BeautifulSoup(reader.read(), 'html5lib')
                text = soup.get_text()
                output = text.split(" ")

                for line in output:
                        for term in COMMERCIAL:
                                if term in line:
                                        commercial_words += 1
                
                doc = doc.split(" ")
        
        return commercial_words/len(doc)

'''
This function counts the number of commercial links present in a webpage
'''
def count_commercial_links(filename, z1):
    with open(filename,encoding="utf-8",errors="ignore") as reader:
        soup = BeautifulSoup(reader.read(), 'html5lib')
        links = Counter([x.get('href') for x in soup.findAll('a')])
        links = links.most_common()
        commercial = 0
        
        for item in links:
            if item[0]: 
                if any(ext in item[0] for ext in COMMERCIAL):
                    commercial += item[1]
    
    return commercial/z1

'''
This function calculates the link-based features
'''
def count_links(filename, z1):
        with open(filename,encoding="utf-8",errors="ignore") as reader:
            soup = BeautifulSoup(reader.read(),'html5lib')
            links = Counter([x.get('href') for x in soup.findAll('a')])
            links = links.most_common()
            total = 0
            external = 0
            contact = 0
            privacy = 0
            p_list = []
            
            for item in links:
                    total += item[1]
                    if item[0]: 
                            if item[0].startswith(('http','ftp','www')):
                                    external += item[1]
                            if any(ext in item[0] for ext in CONTACT):
                                    contact = 1
                            if any(ext in item[0] for ext in PRIVACY):
                                    privacy = 1
                                    
            internal = total - external

        return total/z1, external/z1, internal/z1, contact, privacy # presence of contact and privacy links are boolean features

'''
This function implements the whole casuistic of feature combinations
'''
def features_calc(docs, corpus, vectorizer, features):
    z1 = 200 # empirical observed standardisation value

    for filename, doc in zip(docs, corpus):
        doc_features = []
        
        if features == "link" or features == "comm" or features == "allRem" or features == "allKeep":
            links_counts = count_links(filename, z1)
            doc_features.extend(links_counts)

        if features == "comm" or features == "allRem" or features == "allKeep":
            commercial_links = count_commercial_links(filename, z1)
            commercial_words = count_commercial_keywords(filename, doc)
            doc_features.extend([commercial_links, commercial_words])

        if features == "wordsRem" or features == "wordsKeep" or features == "allRem" or features == "allKeep":
            words = word_features(doc, vectorizer)
            doc_features.extend(words)

        yield doc_features

'''
This function generates the vocabulary for a given corpus
'''
def generate_vocabulary(corpus, min_df):
        vectorizer = CountVectorizer(min_df=min_df)
        vectorizer.fit(corpus)
        return vectorizer

'''
This function normalizes a text to be used as a ML algorithm input
'''
def __normalize_text(line, features):
        line = re.sub('[^a-zA-Z]', ' ', line) # remove punctuations
        line = line.lower() # convert to lowercase
        line = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", line) # remove tags
        line = re.sub("(\\d|\\W)+", " ", line) # remove special char and digits 
        line = line.split() # convert string to list

        if features != "wordsKeep" and features != "allKeep":
                line = [word for word in line if not word in STOPWORDS] # remove stopwords
        
        line = " ".join(line)
        return line

'''
This function extracts clean text from a given HTML file
'''
def preprocess_text(filename, features):
        with open(filename,encoding="utf-8",errors="ignore") as reader:
                soup = BeautifulSoup(reader.read(), 'html5lib') 
                text = soup.get_text()
                output = text.split("\n")
                lines = []
                
                for line in output:
                      line = __normalize_text(line, features)
                      lines.append(line)
                
                doc= " ".join(lines)
                return doc

'''
This function generates an entire clean corpus from HTML files
'''
def generate_corpus(docs, features):
        corpus = []
        
        for doc in docs:
                doc = preprocess_text(doc, features)
                corpus.append(doc)
        
        return corpus

'''
This function loads the CLEF dataset
'''
def data_clef():
        if not os.path.exists('./datasets/CLEF/clef2018collection'):
                print("To perform these experiments you first need to download clef2018collection")
                sys.exit(1)
                
        X = []
        Y = []

        with open('./datasets/CLEF/CLEF2018_qtrust_20180914.txt',newline='') as assestments:
                reader = csv.reader(assestments,delimiter=' ')
                for row in reader:
                        web = row[2]
                        rating = int(row[3])

                        if rating == 0 or rating == 1 or rating == 2 or rating == 3: # relabelling process 
                                for filename in Path('./datasets/CLEF/clef2018collection').rglob(web): # this fucntion finds recursively a file in an entire path
                                        X.append(filename)
                                        break
                                Y.append(1)

                        elif rating == 7 or rating == 8 or rating == 9 or rating == 10: # relabelling process 
                                for filename in Path('./datasets/CLEF/clef2018collection').rglob(web):
                                        X.append(filename)
                                        break
                                Y.append(-1)
      
        return np.array(X), np.array(Y)

'''
This function loads the Schwarz dataset
'''
def data_schwarz():
        df = pd.read_excel("./datasets/Schwarz/web_credibility_relabeled.xlsx")
        ratings = df['Likert Rating']
        urls = df['URL']
        root = os.getcwd()
        path = './datasets/Schwarz/CachedPages'
        os.chdir(path)
        cached_pages_dir = os.getcwd()
        X = []
        Y = []
        
        for url,rating in zip(urls,ratings):
                try:
                        url = url.replace('http://','')
                        url = url.split('/')
                        if url[-1]: # this case deals with urls like 'www.adamofficial.com/us/home'
                                url = '/'.join(url[:-1])
                                os.chdir(url)
                                f = [f for f in os.listdir() if re.match(url[-1]+'*',f) and os.path.isfile(f)]
                        else:
                                url = '/'.join(url)
                                os.chdir(url)
                                f = [f for f in os.listdir() if re.match('index*',f) and os.path.isfile(f)]
                        
                        X.append(os.path.join(os.getcwd(),f[0]))
                        Y.append(rating)
                        os.chdir(cached_pages_dir)

                except:
                        pass
        
        os.chdir(root)
        return np.array(X), np.array(Y)

'''
This function loads the Sondhi dataset
'''
def data_sondhi():
        path1 = './datasets/Sondhi/reliable'
        root = os.getcwd()
        os.chdir(path1)
        arr1 = os.listdir('.')
        path2 = '../unreliable'
        os.chdir(path2)
        arr2 = os.listdir('.')
        X = []
        Y = []
        
        for rel,unrel in zip(arr1,arr2):
                os.chdir('../reliable')
                X.append('./datasets/Sondhi/reliable/'+rel)
                Y.append(-1)
                os.chdir('../unreliable')
                X.append('./datasets/Sondhi/unreliable/'+unrel)
                Y.append(1)
        
        os.chdir(root)
        return np.array(X), np.array(Y)

def start(dataset="Sondhi", features="allKeep", dump="yes", standard=True):

        if dataset == "Sondhi":
                X, Y = data_sondhi()
                n = 5
                min_df = 1

        elif dataset == "Schwarz":
                X, Y = data_schwarz()
                n = 2
                min_df = 0.5

        elif dataset == "CLEF":
                X, Y = data_clef()
                n = 5
                min_df = 0.4

        else:
                print("Unknown dataset")
                sys.exit(1)

        np.random.seed(1) # reproducibility seed
        skf = StratifiedKFold(n_splits=n) # stratified k-fold: preserves the percentage of samples for each class
        ts = str(time.time())
        print("EXPERIMENT ID: ", ts) # we use the timestamp as experiment id

        '''
        For each cost-factor, we perform a n-fold cross validation for the feature set previously selected
        '''
        for cost_factor in range(3):

                accuracies, f1_micro, f1_rel, f1_unrel = [], [], [], []
                it = 1

                for train_index, test_index in skf.split(X,Y):

                        data_train   = X[train_index]
                        corpus_train = generate_corpus(data_train, features)
                        vectorizer = generate_vocabulary(corpus_train, min_df) # for each fold we reset vocabulary associated to training set

                        if dump == "yes":

                                if not os.path.exists('./models'):
                                        os.makedirs('./models')

                                pickle.dump(vectorizer, open("models/vocabulary_"+dataset+"_"+features+"_it"+str(it)+"_cost_fact"+str(cost_factor+1)+"_"+ts+".pkl","wb"))

                        data_train = features_calc(data_train, corpus_train, vectorizer, features)
                        target_train = Y[train_index]

                        if standard:
                                list_data_train = list(data_train)
                                scaler_x = preprocessing.StandardScaler().fit(list_data_train)

                                if dump == "yes":
                                        pickle.dump(scaler_x, open("models/scaler_"+dataset+"_"+features+"_it"+str(it)+"_cost_fact"+str(cost_factor+1)+"_"+ts+".pkl","wb"))

                                data_train = scaler_x.transform(list_data_train)

                        elif not standard:
                                data_train = np.array(list(data_train))
                                nsamples, nx = data_train.shape
                                data_train = data_train.reshape((nsamples, nx))

                        if not os.path.exists('./aux'):
                                os.makedirs('./aux')

                        dump_svmlight_file(data_train, target_train, 'aux/train_'+ts+'.txt')

                        data_test = X[test_index]
                        corpus_test = generate_corpus(data_test, features)
                        data_test = features_calc(data_test, corpus_test, vectorizer, features)
                        target_test  = Y[test_index]

                        if standard:
                                data_test = scaler_x.transform(list(data_test))

                        elif not standard:
                                data_test = np.array(list(data_test))
                                nsamples, nx = data_test.shape
                                data_test = data_test.reshape((nsamples, nx))

                        dump_svmlight_file(data_test, target_test, 'aux/test_'+ts+'.txt')

                        train = svm_parse('aux/train_'+ts+'.txt')
                        aux = svm_parse('aux/test_'+ts+'.txt')
                        test, val = adapt_to_svmlight_format(aux)

                        print("Training it=", it, "cost-factor=", cost_factor+1)

                        model = svmlight.learn(list(train), type='classification', verbosity=0, costratio=cost_factor+1) ## costratio = cost-factor

                        if dump == "yes":
                                svmlight.write_model(model, "models/model_"+dataset+"_"+features+"_it"+str(it)+"_cost_fact"+str(cost_factor+1)+"_"+ts+".dat")

                        predictions = svmlight.classify(model, test)
                        print("Predicting it=", it, "cost-factor=", cost_factor+1)

                        tp, tn, fp, fn = evaluate(predictions)
                        accuracies.append(weighted_accuracy(cost_factor+1,tn,tp,fn,fp)*100)
                        predictions = np.array(predictions)
                        predictions[predictions<0] = -1
                        predictions[predictions>0] = 1
                        f1_micro.append(f1_score(val,predictions,average='micro')) # micro: calculates metrics totally by counting the total true positives, false negatives and false positives
                        cl = f1_score(val, predictions, average=None) # none: returns scores for each class
                        f1_rel.append(cl[0])
                        f1_unrel.append(cl[1])
                        it+=1

                print("The accuracy is", np.mean(accuracies))
                print("The f1-score is", np.mean(f1_micro))
                print("The credible f1-score is", np.mean(f1_rel))
                print("The non-credible f1-score is", np.mean(f1_unrel))
                save_results(dataset, features, cost_factor, ts, accuracies, f1_micro, f1_rel, f1_unrel)

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("dataset", choices=["CLEF", "Sondhi", "Schwarz"]) # DATASETS
        parser.add_argument("features", choices=["link", "comm", "wordsRem", "wordsKeep", "allRem", "allKeep"]) # FEATURE SETS
        parser.add_argument("dump", nargs='?', choices=["yes", "no"], default = 'yes') # DUMP
        args = parser.parse_args()
        dataset = args.dataset
        features = args.features
        dump = args.dump
        standard = True # by default, we apply standard scaler

        # start the training
        start(dataset, features, dump, standard)

