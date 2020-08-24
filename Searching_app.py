import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess

st.title("Movies Searching Engine")
text_input = st.text_input("Enter your interests", "Jenifer aniston david ice")
num_input = st.slider("Choose top", 5, 20, 5)
# Preprocessing
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def lemmatize_stemming(word):
    return stemmer.stem(lemmatizer.lemmatize(word))


def preprocessing(text):
    res = []
    for word in gensim.utils.simple_preprocess(text):
        word = lemmatize_stemming(word)
        if word not in gensim.parsing.preprocessing.STOPWORDS and word not in res:
            res.append(word)
    return res


# Geting
from collections import defaultdict


def reading_params():
    idx2word = defaultdict(str)
    word2idx = defaultdict(int)
    with open("./vocab.txt", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        for line in lines:
            index, word = line.split("<fff>")[0], line.split("<fff>")[1]
            idx2word[index] = word
            word2idx[word] = index
    doc_tfidf = []
    with open("./tfidf.txt", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        for line in lines:
            ones = []
            line = line.split("<fff>")
            index = line[0]
            for ele in line[1:]:
                ele = ele.split(":")
                ones.append(tuple((int(ele[0]), float(ele[1]))))

            doc_tfidf.append(ones)

    return word2idx, idx2word, doc_tfidf


word2idx, idx2word, doc_tfidf = reading_params()

# Searching


def searching(text, num):
    text_processed = preprocessing(text)
    text_processed_id = [word2idx[word] for word in text_processed]
    #doc_tokenized = [[idx2word[ele[0]] for ele in doc] for doc in doc_tfidf]
    score = defaultdict(float)
    for index, a_doc_tfidf in enumerate(doc_tfidf):
        cur_score = 0
        for word_id in text_processed_id:
            for ele in a_doc_tfidf:
                if str(ele[0]) == str(word_id):
                    cur_score += float(ele[1])
        score[index] = cur_score

    A = sorted(score.items(), key=lambda x: x[1], reverse=True)[:num]
    topidx = [ele[0] for ele in A]
    topscore = [ele[1] for ele in A]
    data = pd.read_csv("./netflix_titles.csv")
    top = pd.DataFrame(data=data.loc[topidx]['title'])
    top['score'] = topscore
    st.write("Top " + str(num) + " results")
    st.table(top)


## Running

searching(text_input, num_input)