import streamlit as st
from PIL import Image
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
normalizer = WordNetLemmatizer()
import nltk.data
tokenizer = nltk.data.load('nltk:tokenizers/punkt/PY3/english.pickle')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
stop_words = stopwords.words('english')
custom_stopwords = ["Tesla", "tesla", "TSLA", "tsla", "Rivian", "rivian", "RIVN", "rivn", "NIO", "nio"]
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# NLTK library to build ngrams.
from nltk.util import ngrams
#Library to count words
from collections import Counter
# library to build wordclouds
from wordcloud import WordCloud
# NLTK to analice sentiment. 
#Another Library to perform sentiment analisis.
from textblob import Word, TextBlob
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

# Images for the app:
image_tesla = Image.open('images/Tesla_car.png')
image_nio = Image.open('images/nio_car.png')
image_rivian = Image.open('images/rivian_car.png')

# Streamlit web.
st.set_page_config(page_title="NLP on Two Companies",
        page_icon=("👂"), layout="wide")

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
  
companyA = st.selectbox('Select company A ticker:',
                                  ('None','NIO', 'TSLA', 'RIVN'))
companyB = st.selectbox('Select company B ticker:',
                                  ('None','NIO', 'TSLA', 'RIVN')) 
if companyA and companyB == "None":
    st.write("You must select both companies tickers")
else:      
    st.write('\n')
    st.write('\n') 
    st.subheader("Raw Data") 

    col1, col_mid, col2 = st.columns((1, 0.1, 1))
    with col1:
        if companyA == "TSLA":
            companyA_name = "Tesla"
            st.write(companyA_name)
            companyA_df = scrape("TSLA", 5)
            st.write(companyA_df.head())
        if companyA == "NIO":
            companyA_name = "NIO"
            st.write(companyA_name)
            companyA_df = scrape("NIO", 5)
            st.write(companyA_df.head())
        if companyA == "RIVN":
            companyA_name = "Rivian"
            st.write(companyA_name)
            companyA_df = scrape("RIVN", 5)
            st.write(companyA_df.head())

    with col2:
        if companyA == "TSLA":
            st.image(image_tesla)
            st.write('[https://www.tesla.com/](https://www.tesla.com/)')
        if companyA == "NIO":
            st.image(image_nio)
            st.write('[https://www.nio.com/](https://www.nio.com/)')
        if companyA == "RIVN":
            st.image(image_rivian)
            st.write('[https://rivian.com/](https://rivian.com/)')
           
    st.write('\n')     
    col1, col_mid, col2 = st.columns((1, 0.1, 1))
    with col1:
        if companyB == "TSLA":
            companyB_name = "Tesla"
            st.write(companyB_name)
            companyB_df = scrape("TSLA", 500)
            st.write(companyB_df.head())
        if companyB == "NIO":
            companyB_name = "NIO"
            st.write(companyB_name)
            companyB_df = scrape("NIO", 500)
            st.write(companyB_df.head())
        if companyB == "RIVN":
            companyB_name = "Rivian"
            st.write(companyB_name)
            companyB_df = scrape("RIVN", 500)
            st.write(companyB_df.head())

    with col2:
         if companyB == "TSLA":
            st.image(image_tesla)
            st.write('[https://www.tesla.com/](https://www.tesla.com/)')
         if companyB == "NIO":
            st.image(image_nio)
            st.write('[https://www.nio.com/](https://www.nio.com/)')
         if companyB == "RIVN":
            st.image(image_rivian)
            st.write('[https://rivian.com/](https://rivian.com/)')
    st.markdown("""---""")

    companyA_pt = preprocess_text("".join(companyA_df.tweets), custom_stopwords)
    companyB_pt = preprocess_text("".join(companyB_df.tweets), custom_stopwords)
    st.write('\n')
    st.write('\n')
    wordcloud1 = WordCloud (
                        background_color = "#0E1117",
                        width = 620,
                        height = 410
                            ).generate(' '.join(companyA_pt))
    wordcloud2 = WordCloud (
                        background_color = "#0E1117",
                        width = 620,
                        height = 410
                            ).generate(' '.join(companyB_pt))
    st.header("WordCloud")
    st.write("This Word Cloud is a visual displays of tweets content – text analysis that displays the most prominent or frequent words in the data collected.")  
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.patch.set_facecolor("#0E1117")
    ax1.imshow(wordcloud1, interpolation='bilInear')
    ax1.set_title(companyA, color="white")
    ax1.axis('off')
    ax1.patch.set_facecolor("#0E1117")
    ax2.imshow(wordcloud2, interpolation='bilInear')
    ax2.set_title(companyB, color="white")
    ax2.axis('off')
    ax2.patch.set_facecolor("#0E1117")
    st.pyplot(fig)
    st.markdown("""---""")

    st.subheader("Finding tweets sentiment. Part 1")
    st.write("NLTK already has a built-in, pretrained sentiment analyzer called VADER (Valence Aware Dictionary and sEntiment Reasoner). Since VADER is pretrained, you can get results more quickly than with many other analyzers. Is best suited for language used in social media, like Twitter with short sentences and some slang and abbreviations to classify the sentiment on each tweet, and them plot the sum with the library plotly.") 
    # Clean the data and create a new column with it.
    companyA_df["clean_tweet"] = companyA_df["tweets"].apply(lambda x: clean_text(x, custom_stopwords))
    companyB_df["clean_tweet"] = companyB_df["tweets"].apply(lambda x: clean_text(x, custom_stopwords))
    # Now we can apply the sentiment function and create a new column with it.
    companyA_df["sentiment"] = companyA_df["clean_tweet"].apply(sentiment)
    companyB_df["sentiment"] = companyB_df["clean_tweet"].apply(sentiment)

    # Total sentiment count
    sentiment_count_A =  companyA_df.groupby('sentiment')['sentiment'].count()
    sentiment_count_B = companyB_df.groupby('sentiment')['sentiment'].count()
    #Creating a df with that count to plot. 
    total_sentiments_A = sentiment_count_A.to_frame()
    total_sentiments_A.rename(columns={"sentiment":"count"}, inplace=True)
    total_sentiments_A.reset_index(inplace=True)
    total_sentiments_A["company"] = companyA_name

    total_sentiments_B = sentiment_count_B.to_frame()
    total_sentiments_B.rename(columns={"sentiment":"count"}, inplace=True)
    total_sentiments_B.reset_index(inplace=True)
    total_sentiments_B["company"] = companyB_name

    total_sentiments = [total_sentiments_A, total_sentiments_B]

    total_sentiments = pd.concat(total_sentiments, ignore_index=True)

    # Plotly
    colours = {
        companyA: "#EF3A4C",
        companyB: "#3EC1CD"
    }
    fig = px.histogram(total_sentiments, x="sentiment", y="count",
                color='company', barmode='group',
                height=600,  color_discrete_map=colours)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""---""")
    st.subheader("Finding tweets sentiment. Part 2")
    st.write("In this part I will use the library TextBlob. The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0].  Meaning -1 negative, 0 neutral and +1 positive")

    #building new columns with calculations
    companyA_df["polarity"] = companyA_df["clean_tweet"].apply(lambda x: TextBlob(x).sentiment[0])
    companyA_df["subjectivity"] = companyA_df["clean_tweet"].apply(lambda x: TextBlob(x).sentiment[1])
    companyB_df["polarity"] = companyB_df["clean_tweet"].apply(lambda x: TextBlob(x).sentiment[0])
    companyB_df["subjectivity"] = companyB_df["clean_tweet"].apply(lambda x: TextBlob(x).sentiment[1])

    # Building dataframes for the visualization. 
    companyA_ma = companyA_df[["tweet_date", "polarity"]]
    companyA_ma = companyA_ma.sort_values(by="tweet_date", ascending=True)
    companyA_ma["MA Polarity"] = companyA_ma.polarity.rolling(10, min_periods=3).mean()

    companyB_ma = companyB_df[["tweet_date", "polarity"]]
    companyB_ma = companyB_ma.sort_values(by="tweet_date", ascending=True)
    companyB_ma["MA Polarity"] = companyB_ma.polarity.rolling(10, min_periods=3).mean()

    #Plotting both graph with Plotly. 
    fig1 = px.line(companyA_ma, x="tweet_date", y="MA Polarity", title= companyA_name + " polarity moving average for today")
    fig1.update_traces(line_color='#EF3A4C')
    fig1.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(companyB_ma, x="tweet_date", y="MA Polarity", title= companyB_name + " polarity moving average for today")
    fig2.update_traces(line_color='#3EC1CD')
    fig2.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Visualizing Subjectivity and Polarity")
    st.write("Subjectivity quantifies the amount of personal opinion and factual information contained in the text. The higher subjectivity means that the text contains personal opinion rather than factual information.")

    trace1 = go.Scatter(x = companyA_df["polarity"], y = companyA_df["subjectivity"], name=companyA_name, marker=dict(color='rgb(34,163,192)'), mode='markers')
    trace2 = go.Scatter(x = companyB_df["polarity"], y = companyB_df["subjectivity"], name=companyB_name, mode='markers')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(trace1)
    fig.add_trace(trace2,secondary_y=True)
    fig['layout'].update(height = 900, width = 700, title = "Polarity and subjectivity scatter plot", xaxis_title="Polarity", yaxis_title="Subjectivity")
    st.plotly_chart(fig)

    st.markdown("""---""")
    st.subheader("Counting the most used words by sentiment. ")
        # Joining and preprocessing all tweets based on sentiment.
    companyA_preprocess_positive = preprocess_text("".join(companyA_df["tweets"][companyA_df["sentiment"] == "positive"]), custom_stopwords)
    companyA_preprocess_negative = preprocess_text("".join(companyA_df["tweets"][companyA_df["sentiment"] == "negative"]), custom_stopwords)

    companyB_preprocess_positive = preprocess_text("".join(companyB_df["tweets"][companyB_df["sentiment"] == "positive"]), custom_stopwords)
    companyB_preprocess_negative = preprocess_text("".join(companyB_df["tweets"][companyB_df["sentiment"] == "negative"]), custom_stopwords)

    #Counting the words of each sentiment. 
    companyA_bow_positive = Counter(companyA_preprocess_positive)
    companyA_bow_negative = Counter(companyA_preprocess_negative)

    companyB_bow_positive = Counter(companyB_preprocess_positive)
    companyB_bow_negative = Counter(companyB_preprocess_negative)

    # Printing the most common words by sentiment. 
    col1, col_mid, col2 = st.columns((1, 0.1, 1))
    with col1:
        st.write("Positive words  for " + companyA_name +".")
        most_common_companyA_positive = companyA_bow_positive.most_common(10)
        st.write(str(most_common_companyA_positive))

        st.write("Negative words for " + companyA_name +".")
        most_common_companyA_negative = companyA_bow_negative.most_common(10)
        st.write(str(most_common_companyA_negative))

    with col2:
        st.write("Positive words  for " + companyB_name +".")
        most_common_companyB_positive = companyB_bow_positive.most_common(10)
        st.write(str(most_common_companyB_positive))

        st.write("Negative words for " + companyB_name +".")
        most_common_companyB_negative = companyB_bow_negative.most_common(10)
        st.write(str(most_common_companyB_negative))

    st.markdown("""---""")

    st.subheader("This is the end of this project in which I have some limitations, the free tweeter API doesn't let me collect tweets selecting specific dates from which the comparison will be more precise in time. Positive and negative word count doesn't give much insight, so in next projects N-grams will be incorporated to gain more context about those words and feelings.")
