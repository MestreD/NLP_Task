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
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = stopwords.words('english')
custom_stopwords = ["Tesla", "tesla", "TSLA", "tsla", "Rivian", "rivian", "RIVN", "rivn"]
normalizer = WordNetLemmatizer()
#Library to count words
from collections import Counter
# library to build wordclouds
from wordcloud import WordCloud
# NLTK to analice sentiment. 
import nltk
nltk.download('vader_lexicon')
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
import plotly.io as pio
pio.renderers.default='browser'
# Library to handle dates in different formats. 
import datetime


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
from functions import scrape, get_part_of_speech, preprocess_text, clean_text, wordcloud, sentiment

# Streamlit web.
st.set_page_config(page_title="NLP on Two Companies",
        page_icon="chart_with_upwards_trend", layout="wide")

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()


with header:
    st.title("NLP analysis on two companies from twitter posts data.")
    st.write("This project is an approach to apply Natural Language Processing on twitter posts using tweepy API, to gain insight of the topics that interest the most people, public opinion and sentiment analysis about two different companies in the Electric Vehicles industrial sector.")
    st.write('\n')
    


with dataset:
    st.header("Data Acquisition")  
    st.write("Tweepy is an open-source python package to access the Twitter API. Using this package, we can retrieve tweets of users, retweets etc. In our project, we will use this package to get live tweets based on two given search string and limiting the data extraction specifying the number of tweets desired")
    st.markdown("""---""")
  
companyA = st.selectbox('Please select company A ticker:',
                                  ('None','NIO', 'TSLA', 'RIVN'))
companyB = st.selectbox('Please select company B ticker:',
                                  ('None','NIO', 'TSLA', 'RIVN'))       
st.write('\n')
st.write('\n') 
st.subheader("Raw Data") 

col1, col_mid, col2 = st.columns((1, 0.1, 1))
with col1:
    if companyA == "TSLA":
        st.write("Tesla")
        st.write(scrape("TSLA", 100).head())
    if companyA == "NIO":
        st.write("NIO")
        st.write(scrape("NIO", 100).head())
    if companyA == "RIVN":
        st.write("Rivian")
        st.write(scrape("NIO", 100).head())

with col2:
    if companyA == "TSLA":
        st.write('[https://www.tesla.com/](https://www.tesla.com/)')
        st.write(f'<iframe \
                     width="400" \
                     height="300"\
                     src="https://en.wikipedia.org/wiki/Tesla,_Inc."></iframe>',
                     unsafe_allow_html=True )   
    if companyA == "NIO":
        st.write('[https://www.nio.com/](https://www.nio.com/)')
        st.write(f'<iframe \
                     width="400" \
                     height="300"\
                     src="https://en.wikipedia.org/wiki/NIO_ES8."></iframe>',
                     unsafe_allow_html=True )  
    if companyA == "RIVN":
        st.write('[https://rivian.com/](https://rivian.com/)')
        st.write(f'<iframe \
                     width="400" \
                     height="300"\
                     src="https://en.wikipedia.org/wiki/Rivian"></iframe>',
                     unsafe_allow_html=True )  
st.write('\n')     
col1, col_mid, col2 = st.columns((1, 0.1, 1))
with col1:
    if companyB == "TSLA":
        st.write("Tesla")
        st.write(scrape("TSLA", 100).head())
    if companyB == "NIO":
        st.write("NIO")
        st.write(scrape("NIO", 100).head())
    if companyB == "RIVN":
        st.write("Rivian")
        st.write(scrape("NIO", 100).head())

with col2:
    if companyB == "TSLA":
        st.write('[https://www.tesla.com/](https://www.tesla.com/)')
        st.write(f'<iframe \
                     width="400" \
                     height="300"\
                     src="https://en.wikipedia.org/wiki/Tesla,_Inc."></iframe>',
                     unsafe_allow_html=True )   
    if companyB == "NIO":
        st.write('[https://www.nio.com/](https://www.nio.com/)')
        st.write(f'<iframe \
                     width="400" \
                     height="300"\
                     src="https://en.wikipedia.org/wiki/NIO_ES8."></iframe>',
                     unsafe_allow_html=True )  
    if companyB == "RIVN":
        st.write('[https://rivian.com/](https://rivian.com/)')
        st.write(f'<iframe \
                     width="400" \
                     height="300"\
                     src="https://en.wikipedia.org/wiki/Rivian"></iframe>',
                     unsafe_allow_html=True ) 
st.markdown("""---""")

