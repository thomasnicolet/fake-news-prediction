# based on 
# https://towardsdatascience.com/predicting-fake-news-using-nlp-and-machine-learning-scikit-learn-glove-keras-lstm-7bbd557c3443
import pandas as pd
import numpy as np
import contractions 
import re
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import time 
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from multiprocessing import Pool

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['figure.figsize'] = [10, 10]
import seaborn as sns
sns.set_theme(style="darkgrid")


def vizualize_label_distribution(df):
    # Vizualize distribution 
    sns.countplot(x='label', data=df, palette='Set3')
    plt.title("Number of Fake and Genuine News after dropping missing values")
    plt.show()

def vizualize_content_length(df):
    # Now lets look at distribution of content length
    sns.boxplot(y='raw_content_length', x='label', data=df, palette="Set3", showmeans=True)
    plt.title("Distribution of content length")
    #plt.ylim(0, 15000) # Uncomment to zoom in
    plt.show()

def word_cloud(df, genuine = True):

    # join all texts in resective labels, 0 being fake and 1 genuine
    all_texts_gen = " ".join(df[df['label']==0]['content_joined'])
    all_texts_fake = " ".join(df[df['label']==1]['content_joined'])

    stopwords = set(nltk.corpus.stopwords.words('english'))

    if genuine:
        # Wordcloud for Genuine News
        wordcloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 10).generate(all_texts_gen)                       
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()
    else: 
        # Worldcloud for Fake News
        wordcloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 10).generate(all_texts_fake)                    
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        
        plt.show()

def get_average_word_length(seq_word_tokens):
    return np.mean([len(word) for seq in seq_word_tokens for word in seq])

# Count the avg number of words in each sentence
def get_average_words_in_sent(seq_word_tokens):
    return np.mean([len(seq) for seq in seq_word_tokens])

# Clean the punctuations
def get_seq_tokens_cleaned(seq_tokens):
    no_punc_seq = [each_seq.translate(str.maketrans('', '', string.punctuation)) for each_seq in seq_tokens]
    sent_word_tokens = [word_tokenize(each_sentence) for each_sentence in no_punc_seq]
    return sent_word_tokens

def format_df_copy(df):
    # Get sentence tokens, i.e. list where elemns are sentences
    df['sent_tokens'] = df['content'].apply(sent_tokenize)

    # number of sentences in fake and genuine article
    df['len_sentence'] = df['sent_tokens'].apply(len)

    df['sent_word_tokens'] = df['sent_tokens'].apply(lambda x: [word_tokenize(each_sentence) for each_sentence in x])

    df['sent_word_tokens'] = df['sent_tokens'].apply(lambda x: get_seq_tokens_cleaned(x))
    df['avg_words_per_sent'] = df['sent_word_tokens'].apply(lambda x: get_average_words_in_sent(x))
    df['avg_word_length'] = df['sent_word_tokens'].apply(lambda x: get_average_word_length(x))

    return df

def boxplot_number_of_sentences(df, yMin=None, yMax=None):

    # Lets check number of sentences in real and fake articles
    sns.boxplot(y='len_sentence', x='label', data=df, palette="Set3")
    plt.title("Boxplot of Number of Sentences in Fake and Genuine Articles")
    #plt.ylim(0, 100) # Uncomment to zoom in
    plt.ylim(yMin, yMax) 
    plt.show()

def boxplot_average_number_of_words_in_sentences(df, yMin=None, yMax=None):
    # show average number of words per sentence
    sns.boxplot(y='avg_words_per_sent', x='label', data=df, palette="Set3")
    plt.title("Boxplot of the Average Number of Words per Sentence in Fake and Genuine Articles")
    plt.ylim(yMin, yMax) 
    plt.show()

def boxplot_average_word_length(df, yMin=None, yMax=None):
    sns.boxplot(y='avg_word_length', x='label', data=df, palette="Set3")
    plt.title("Boxplot of the Average word length in Fake and Genuine Articles")
    plt.ylim(yMin, yMax) 
    plt.show()


