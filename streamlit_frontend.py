import streamlit as st
# Pandas: This library will used to create and work with dataframes.
import pandas as pd 
# Matplotlib: Used to plot graphics. 
import matplotlib.pyplot as plt
# Seaborn: Used to better style and improve graphics. 
import seaborn as sns
# Numpy for mathematic calculations. 
import numpy as np
#Tweepy: Used to work with the Twitter API.
import tweepy as tw
#ConfigParser: Used to get credentials for the twitter API.
import configparser as cp
#NLTK its used to pre-proccess text and regex its used to filter that text.
import nltk, re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
stop_words = stopwords.words('english')
custom_stopwords = ["Tesla", "tesla", "TSLA", "tsla", "Rivian", "rivian", "RIVN", "rivn"]
normalizer = WordNetLemmatizer()
#Library to count words
from collections import Counter
# library to build wordclouds
from wordcloud import WordCloud
# NLTK to analice sentiment. 
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
#Another Library to perform sentiment analisis.
from textblob import Word, TextBlob
# NLTK library to build ngrams.
from nltk.util import ngrams
# NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
import networkx as nx
from pyvis.network import Network
# Built on top of plotly.js, plotly.py is a high-level, declarative charting library.
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Importing and authenticating API credentials from the config file. 
config =  cp.ConfigParser()
config.read("config.ini")

api_key = config["twitter"]["api_key"]
api_key_secret = config["twitter"]["api_key_secret"]
access_token = config["twitter"]["access_token"]
access_token_secret = config["twitter"]["access_token_secret"]

# Authentication

auth = tw.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tw.API(auth)

# Functions to work with in this project:


# Function to perform data extraction from twitter.
def scrape(words, numtweet):
# We are using .Cursor() to search
# through twitter for the required tweets.
# The number of tweets can be
# restricted using .items(number of tweets)
    tweets = tw.Cursor(api.search_tweets,
                               words, 
                               lang="en",
                               tweet_mode='extended').items(numtweet)


# .Cursor() returns an iterable object. Each item in
# the iterator has various attributes
# that you can access to
# get information about each tweet
    list_tweets = [tweet for tweet in tweets]
 
# we will iterate over each tweet in the
# list for extracting information about each tweet
    columns=['tweet_date','tweets']
    data = []
    for tweet in list_tweets:
        tweet_date = tweet.created_at
# Retweets can be distinguished by
# a retweeted_status attribute,
# in case it is an invalid reference,
# except block will be executed
        try:
            tweets = tweet.retweeted_status.full_text
        except AttributeError:
            tweets = tweet.full_text
            data.append([tweet_date, tweets])
# Creating DataFrame using pandas
    df = pd.DataFrame(data, columns=columns)
    print(words + " Data.")
    return df 


# Streamlit web.

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

with header:
    st.title("NLP analysis on two companies from twitter posts data.")
    st.markdown("<h8 style='text-align: justify;'>This project is an approach to apply Natural Language Processing on twitter posts using tweepy API, to gain insight of the topics that interest the most people, public opinion and sentiment analysis about two different companies in the Electric Vehicles industrial sector.</h8>", unsafe_allow_html=True)

with dataset:
    st.header("Data Acquisition")  
    st.markdown("<h8 style='text-align: justify;'>Tweepy is an open-source python package to access the Twitter API. Using this package, we can retrieve tweets of users, retweets etc. In our project, we will use this package to get live tweets based on two given search string and limiting the data extraction specifying the number of tweets desired.</h8>", unsafe_allow_html=True)
    tesla = scrape("TSLA", 100)
    rivian = scrape("RIVN", 100)
    st.markdown("<h3 style='text-align: centered; color=red'> Raw Data </h3>", unsafe_allow_html=True)
    st.markdown("<h11 style='text-align: centered; color=red'> Tesla </h11>", unsafe_allow_html=True)
    st.write(tesla.head())
    st.markdown("<h11 style='text-align: centered; color=red'> Rivian </h11>", unsafe_allow_html=True)
    st.write(rivian.head())