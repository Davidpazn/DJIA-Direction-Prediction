#!/usr/bin/env python3
# coding: utf-8

__author__ = "David Pacheco Aznar"
__email__ = "david.marketmodels@gmail.com"

# The aim of this script is to write the core functions that will be needed in 
# order to perform our nlp tasks. This way, notebooks will be cleaner.

import subprocess
import sys

# installation of package extras
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

def lang_install(module):
    subprocess.check_call([sys.executable, "-m", "spacy", "download", module])

# string manipulation imports
import re
import string
# fix contractions
install('contractions')
import contractions

# manipulate series if necessary
import pandas as pd

# gensim
import gensim
from gensim import corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess

# spacy
# install the requirements:
install('spacy')
lang_install('en_core_web_sm')

import spacy
nlp = spacy.load('en_core_web_sm')  # load small size model for nlp task
STOP_WORDS = nlp.Defaults.stop_words


# ##############################################################################
# ############################## DATA CLEANING #################################
# ##############################################################################

def clean_news(news, lemma_flag=False, 
               stop=True, remove_keywords=False, 
               emojis=False, 
               postags=['NOUN', 'VERB', 'ADV', 'ADJ', 'PROPN'], # spacy postags
               min_token_len=3, bigram_min=5, bigram_threshold=100,   # gensim bigrams
               deacc=True  # simple preprocess deaccent Ŝ -> S
               ) -> str:

    # create list of news
    news_list = news.values.tolist()

    # remove unicode default encoding.
    news_list = [re.sub(r'\bsay\S+\b', '', nw) for nw in news_list]

    # remove emails, newlines and single quotes
    news_list = [re.sub(r'\s+', ' ', nw) for nw in news_list]
    news_list = [re.sub(r'\S*@\S*\s?', '', nw) for nw in news_list]

    # remove traces of byte strings: b' or b"
    news_list = [re.sub(r"b\'", "", nw) for nw in news_list]
    news_list = [re.sub(r'b\"', "", nw) for nw in news_list]
    
    # remove quoting
    news_list = [re.sub(r"\'", "", nw) for nw in news_list]

    # news = news.lower()  # lower all letters

    # remove special characters (UNICODE)
    if not emojis:
        emoji = re.compile(
          '['
          u"\U0001F600-\U0001F64F"  # emoticons
          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
          u"\U0001F680-\U0001F6FF"  # transport & map symbols
          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
          u"\U00002500-\U00002BEF"  # chinese char
          u"\U00002702-\U000027B0"
          u"\U000024C2-\U0001F251"
          u"\U0001f926-\U0001f937"
          u"\U00010000-\U0010ffff"
          u"\u2640-\u2642"
          u"\u2600-\u2B55"
          u"\u200d"
          u"\u23cf"
          u"\u23e9"
          u"\u231a"
          u"\ufe0f"  # dingbats
          u"\u3030"
          ']+',
          flags=re.UNICODE
        )  # can also use no SYM from SpaCy.
        news_list = [emoji.sub(r'', nw) for nw in news_list]

    # Tokenize:
    word_list = [simple_preprocess(nw, deacc=deacc, min_len=min_token_len) for nw in news_list]    

    # remove stop words
    if stop:
        word_nostops = [[w for w in nw if w not in STOP_WORDS] for nw in word_list]
    
    else:
        word_nostops = word_list
    # N-GRAMS
    bigrams, trigrams, ngrams_df = detect_ngrams(corpus=word_nostops)

    lst_corpus = preprocess_ngrams(word_nostops, ngrams=1, grams_join=" ",
                                   lst_ngrams_detectors=[bigrams, trigrams])

    # Lemmatisation: Convert words into root of word
    word_list_lemma_ = lemmatize(lst_corpus, postags=postags)


    return word_list_lemma_, ngrams_df


def lemmatize(word_list, postags):
    lemma_texts = []
    for txt in word_list:
        s = nlp(" ".join(txt))
        lemma_texts.append([token.lemma_ for token in s if token.pos_ in postags])
    return lemma_texts


def preprocess_ngrams(lst_corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[]):
    # detect common bi-grams and tri-grams
    if len(lst_ngrams_detectors) != 0:
        for detector in lst_ngrams_detectors:
            lst_corpus = list(detector[lst_corpus])
    return lst_corpus


def detect_ngrams(corpus, grams_join=" ", min_count=5):
    # Fit Models
    lst_corpus = preprocess_ngrams(corpus, ngrams=1, grams_join=grams_join)
    
    # detect bigrams
    bigrams_detector = Phrases(lst_corpus, delimiter=grams_join.encode(),
                               min_count=min_count, threshold=min_count * 2
                               )
    bigrams_detector = Phraser(bigrams_detector)

    # detect trigrams
    trigrams_detector = Phrases(bigrams_detector[lst_corpus], 
                                delimiter=grams_join.encode(), min_count=min_count, 
                                threshold=min_count * 2
                                )

    trigrams_detector = Phraser(trigrams_detector)

    df_ngrams = pd.DataFrame([{"word": [ki.decode('utf-8') for ki in k], "freq": v} for k, v in
                               trigrams_detector.phrasegrams.items()])
    df_ngrams["ngrams"] = df_ngrams["word"].apply(lambda x: x.count(grams_join) + 1)
    df_ngrams = df_ngrams.sort_values(["ngrams", "freq"], ascending=[True, False])

    return bigrams_detector, trigrams_detector, df_ngrams


def apply_ngrams(news, ngrams):
    from tqdm import tqdm
    ngram_words_list = []
    for word_list in tqdm(news):
        for gram in ngrams:    
            for ix in range(len(word_list)-1):
                if ix < len(word_list)-1:
                    if word_list[ix] == gram[0] and word_list[ix + 1] == gram[1]:
                        # print(" ".join(word_list[ix: ix+2]))
                        word_list[ix] = [" ".join(word_list[ix: ix+2])]
                        word_list.pop(ix+1)
        ngram_words_list.append([x[0] if (type(x) == list) else x for x in word_list])
    return ngram_words_list











