import re
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

punctuation = r"[\"\#\$\%\&\\'\(\)\*\+,\-/:<=>@\[\\\]\^_\{\|\}\~]"

sentence_tokenizer = r"[\.\?!;\s+]"

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def def_value():
    return 0
    
def cleanup(df):
    df.text = df.text.str.lower()
    df.text = df.text.str.replace(punctuation, ' ')
    df.text = df.text.str.split(sentence_tokenizer)
    df.text = df.text.apply(lambda x: [item for item in x if item not in stop_words])
    return df

def vocab(df):
    V = set()
    for element in df['text'].values:
        for word in element:
            V.add(word)
    return V            

def wordcount(df):
    D = defaultdict(def_value)
    for i in df['text'].values:
        for j in i:
            if j not in D.keys():
                D[j] = 1
            elif j in D.keys():
                D[j] += 1    
    return D

def train(df):
    N = df['text'].size
    
    V = vocab(df)
    N_V = len(V)
    
    Neg = df[df['label'] == '0']
    Pos = df[df['label'] == '1']
    Neg_size = Neg['text'].size
    Pos_size = Pos['text'].size

    D_POS = wordcount(Pos)
    D_NEG = wordcount(Neg)

    pri_pos_prob = np.log(Pos_size/N)
    pri_neg_prob = np.log(Neg_size/N)

    prob_pos = {}
    prob_neg = {}
    for word in V:   
        prob_pos[word] = np.log((D_POS[word]+1)/(Pos_size+N_V))
        prob_neg[word] = np.log((D_NEG[word]+1)/(Neg_size+N_V))
    return (V, pri_pos_prob, pri_neg_prob, prob_pos, prob_neg)    

def test_helper(line,V,pri_pos_prob, pri_neg_prob, prob_pos, prob_neg):
    test_pos = pri_pos_prob
    test_neg = pri_neg_prob
    for word in line:
         if word in V:
             test_pos += prob_pos[word]
             test_neg += prob_neg[word]        
    if test_pos>test_neg:
        return '1'
    else:
        return '0'                 


def test(df, V, pri_pos_prob, pri_neg_prob, prob_pos, prob_neg):
    result = []
    for line in df['text'].values:
        result.append(test_helper(line,V, pri_pos_prob, pri_neg_prob, prob_pos, prob_neg))
    return result

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0      

if __name__ == "__main__":
    #pd.set_option('display.max_colwidth', 5000)
    df1 = pd.read_csv("Train.csv", dtype = 'string')
    df1_clean = cleanup(df1)
    train_data = train(df1_clean)
    df2 = pd.read_csv("Valid.csv", dtype = 'string')
    df2_clean = cleanup(df2)     
    result = test(df2_clean, train_data[0], train_data[1], train_data[2], train_data[3], train_data[4])
    print(result)
    print(accuracy_metric(df2['label'].values,result))  
