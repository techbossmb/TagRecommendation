import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from utils import clean_text, sort_score, map_topk_words
import os
from nltk.corpus import wordnet
import pickle

'''
Description: use Term-Frequency - Inverse DocumentFrequency (TFIDF) to identify important words in a text
and suggest words simialar to the TF-IDF identified words. 
@use-case: A tag recommendation system based on importance of words in a document
@author: techbossmb
@date: 02/23/19
'''

datapath = '..{}data'.format(os.sep)
train_file = '{}{}stackoverflow-data-idf.json'.format(datapath, os.sep)
test_file = '{}{}stackoverflow-test.json'.format(datapath, os.sep)

def load_jsondata(filepath):
    df = pd.read_json(filepath, lines=True)
    df['body'] = df['body'].apply(lambda x:clean_text(x))
    data = df['body'].tolist()
    return data

def build_tfidf_model(train_file):
    train_data = load_jsondata(train_file)
    # get word count based on train data
    count_vectorizer = CountVectorizer(max_df=0.8,stop_words='english')
    word_counts = count_vectorizer.fit_transform(train_data)
    # build tf-idf model
    tf_idf = TfidfTransformer(smooth_idf=True, use_idf=True)
    tf_idf.fit(word_counts)
    return tf_idf, count_vectorizer

def get_topk_words(count_vectorizer, tf_idf, feature_names, text, k=3):
    test_wordcount = count_vectorizer.transform([text])
    test_tf_idf = tf_idf.transform(test_wordcount)
    sorted_score = sort_score(test_tf_idf.tocoo())
    topkwords = map_topk_words(feature_names, sorted_score, k)
    return topkwords

def estimate_tfidf_on_testdata(count_vectorizer, tf_idf, test_data, k):
    feature_names = count_vectorizer.get_feature_names()
    topkwords_list = []
    for text in test_data:
        topkwords = get_topk_words(count_vectorizer, tf_idf, feature_names, text, k)
        topkwords_list.append(topkwords)
    return topkwords_list

def get_similar_words(word):
    similar_words = wordnet.synsets(word)
    recommendations = [word]
    for similar_word in similar_words:
        recommendations.append(similar_word.lemmas()[0].name())
    recommendations = set(recommendations)
    return recommendations
    
if __name__=='__main__':
    tfidf_path = '{}{}tfidf.pk'.format(datapath, os.sep)
    countvector_path = '{}{}countvector.pk'.format(datapath, os.sep)

    # load saved (trained) model or train new model if it doesn't exist
    if os.path.exists(tfidf_path) and os.path.exists(countvector_path):
        print('loading saved model')
        tf_idf = pickle.load(open(tfidf_path, 'rb'))
        count_vectorizer = pickle.load(open(countvector_path, 'rb'))
    else:
        print('building tf-idf model')
        tf_idf, count_vectorizer = build_tfidf_model(train_file)
        pickle.dump(tf_idf, open(tfidf_path, "wb"))
        pickle.dump(count_vectorizer, open(countvector_path, "wb"))

    test_data = load_jsondata(test_file)
    num_of_tags = 10
    topkwords_list = estimate_tfidf_on_testdata(count_vectorizer, tf_idf, test_data, num_of_tags)
    test_index = 0
    print('Text :{}'.format(test_data[test_index]))
    print('Top {} words from text'.format(num_of_tags))
    topkwords = topkwords_list[test_index]
    # show important words in the document plus other words similar to it (recommended tags)
    for topkword in topkwords:
        print('word: {}, tf-idf score: {}, recommended tags: {}'.format(topkword, \
                                                                topkwords[topkword], \
                                                                get_similar_words(topkword)))
