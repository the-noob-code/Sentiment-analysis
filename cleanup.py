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
