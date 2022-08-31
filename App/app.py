# Core Pkgs
import re
import os
from time import sleep
import base64
import altair as alt
import requests
import matplotlib.pyplot as plt
import numpy as np
# EDA Pkgs
import pandas as pd
import seaborn as sns
import streamlit as st
import tweepy
from PIL import Image
from textblob import TextBlob
import joblib
from about import *
from dotenv import load_dotenv

load_dotenv()
# Twitter Api creation
consumerKey = os.environ.get('consumerKey')
consumerSecret=os.environ.get('consumerSecret')
accessToken=os.environ.get('accessToken')
accessTokenSecret=os.environ.get('accessTokenSecret')

try:
    # Create the authentication object
    authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret))

    # Set the access token and access token secret
    authenticate.set_access_token(accessToken, accessTokenSecret))
    print(authenticate)
    # Creating the API object while passing in auth information
    api = tweepy.API(authenticate, wait_on_rate_limit=True)
except Exception as e:
    st.subheader("Error Encountered:{}".format(e))


def sidebar_bg(side_bg):
    side_bg_ext = 'png'

    st.markdown(
        f"""
      <style>
      .stApp {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
           background-size: cover
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )




pipe_lr = joblib.load(
    open("models/emotion_classifier_pipe_lr_03_june_2021.pkl",
         "rb"))


# Fxn
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐",
                       "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}


def Show_Recent_Tweets(raw_text):
    # Extract 100 tweets from the twitter user
    try:
        posts = api.user_timeline(screen_name=raw_text,lang='en', count=1, tweet_mode="extended")
    except tweepy.TweepyException as e:
        st.warning("Tweepy error:{}".format(e))
        ph = st.empty()
        N = 10
        bar=st.progress(0)
        for secs in range(0, N, 1):
            mm, ss = (N-secs) // 60, (N-secs) % 60
            bar.progress((secs+1)*10)
            ph.metric("Redirecting in...", f"{mm:02d}:{ss:02d}")
            sleep(1)

        st.experimental_rerun()

    def get_tweets():
        l = []
        i = 1
        for tweet in posts[:1]:
            l.append(tweet.full_text)
            i = i + 1
        return l

    recent_tweets = get_tweets()
    return recent_tweets


# Main Application
count1 = 0
count2 = 0

def main():
   # sentiment_dispaly("Twitter Sentiment Analysis App",500,500)
    url1 = "https://miro.medium.com/max/1400/1*FKQD-ZkhRbS3Q-MC6kybOg.png"
    image1 = Image.open(requests.get(url1, stream=True).raw)
    st.image(image1, width=500, channels="RGB", output_format="auto")
    side_bg = 'App/depositphotos_3506443-stock-illustration-criticism-word-collage-on-black.jpg'
    sidebar_bg(side_bg)
    st.title("Twitter Sentiment Analysis  App")
    menu = ["Sentiment analysis","query", "About"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Sentiment analysis":
        # add_page_visited_details("Home",datetime.now())
        st.subheader("sentiment analysis  from twitter")

        with st.form(key='emotion_clf_form',clear_on_submit=True):
            raw_text1 = st.text_area("Enter the exact twitter handle of the Personality (without @)",key='user')
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            # Apply Fxn Here
            raw_text = Show_Recent_Tweets(raw_text1)
            raw_text0 = raw_text[:]
            raw_text = ''.join(map(str, raw_text))
            # lang=detect(raw_text)
            # print(lang)
            # if(lang!='en'):
            #     translator = google_translator()
            #     raw_text=translator.translate(raw_text,lang_tgt='en')
            #     global count2
            #     count2=1



            def cleanTxt(text):
                text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
                text = re.sub('#', '', text)  # Removing '#' hash tag
                text = re.sub('RT[\s]+', '', text)  # Removing RT
                text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink

                return text
            raw_text=cleanTxt(raw_text)
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            with st.spinner(f'analysing and searching for {raw_text1}.....'):
                sleep(10)
                with col1:
                    username = str(raw_text1)
                    try:
                        user = api.get_user(screen_name=username)
                    except Exception as e:
                        st.warning("Error:{}".format(e))
                        ph = st.empty()
                        N = 10
                        bar = st.progress(0)
                        for secs in range(0, N, 1):
                            mm, ss = (N - secs) // 60, (N - secs) % 60
                            bar.progress((secs + 1) * 10)
                            ph.metric("Redirecting in...", f"{mm:02d}:{ss:02d}")
                            sleep(1)

                        st.experimental_rerun()

                    st.success("Profile Picture")
                    url = 'https://unavatar.io/twitter/' + str(raw_text1)
                    print(user)
                    im = Image.open(requests.get(url, stream=True).raw)
                    st.image(im, caption=user.name, width=200, channels="RGB", output_format="auto")

                    st.write("{} id:{}".format(user.name, user.id))
                    st.write("{} Follower:{}".format(user.name, user.followers_count))
                    if len(user.description) > 0:
                        st.write("Description:{}".format(user.description))
                    else:
                        pass
                    st.success("Original Tweet")
                    st.write(raw_text0)
                    # st.success("Cleaned Tweet")
                    # st.write(raw_text)
                    st.success("Prediction")
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{}:{}".format(prediction, emoji_icon))
                    st.write("Confidence:{}".format(np.max(probability)))

                with col2:
                    st.success("Prediction Probability")
                    st.write(probability)
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    st.write(proba_df.T)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["sentiment", "probability"]

                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='sentiment', y='probability', color='sentiment')
                    st.altair_chart(fig, use_container_width=True)
                    st.write("polarity:")
                    blob = TextBlob(raw_text)
                    result_sentiment = blob.sentiment
                    st.success(result_sentiment)
                    global count1
                    count1 = 1
                if count1 == 1:
                    def Plot_Analysis():

                        st.success("Generating Visualisation for Sentiment Analysis")
                        st.set_option('deprecation.showPyplotGlobalUse', False)

                        posts = api.user_timeline(screen_name=raw_text1, count=100, lang='en',tweet_mode="extended")

                        df = pd.DataFrame(
                            data=[[len(tweet.full_text), tweet.full_text, tweet.created_at] for tweet in posts],
                            columns=['Length Of Tweet', 'Tweets', 'Tweet Date'])

                        # Create a function to clean the tweets
                        def cleanTxt(text):
                            text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
                            text = re.sub('#', '', text)  # Removing '#' hash tag
                            text = re.sub('RT[\s]+', '', text)  # Removing RT
                            text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink

                            return text

                        # Clean the tweets
                        # def lang_check(text):
                        #     lang = detect(text)
                        #     if (lang != 'en'):
                        #         translator = google_translator()
                        #         text = translator.translate(text, lang_tgt='en')
                        #     else:
                        #         text=text
                        #     return text
                        # df['Tweets']=df['Tweets'].apply(lang_check)
                        df['Tweets'] = df['Tweets'].apply(cleanTxt)

                        def getSubjectivity(text):
                            return TextBlob(text).sentiment.subjectivity

                        # Create a function to get the polarity
                        def getPolarity(text):
                            return TextBlob(text).sentiment.polarity

                        # Create two new columns 'Subjectivity' & 'Polarity'
                        df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
                        df['Polarity'] = df['Tweets'].apply(getPolarity)

                        def getAnalysis(score):
                            if score < 0:
                                return 'Negative'
                            elif score == 0:
                                return 'Neutral'
                            else:
                                return 'Positive'

                        df['Analysis'] = df['Polarity'].apply(getAnalysis)
                        df['Sentiment'] = df['Tweets'].apply(predict_emotions)

                        return df

                    st.markdown('##')
                    st.markdown('##')
                    st.markdown('##')
                    df = Plot_Analysis()
                    st.write(sns.countplot(x=df["Analysis"], data=df))
                    st.pyplot(use_container_width=True)
                    st.markdown('##')
                    st.markdown('##')
                    st.bar_chart(df['Analysis'])
                    st.markdown('##')
                    st.markdown('##')
                    st.write(df)
                    st.markdown('##')
                    st.markdown('##')
                    # plot the polarity and subjectivity
                    fig = plt.figure(figsize=(8, 6))
                    for i in range(0, df.shape[0]):
                        plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Blue')
                    plt.title('Sentiment Analysis')
                    plt.xlabel('Polarlity')
                    plt.ylabel('Subjectivity')
                    st.balloons()
                    st.pyplot(fig)
                    st.markdown('##')
                    st.markdown('##')
                    # get the percentage of positive tweets
                    ptweets = df[df.Analysis == 'Positive']
                    ptweets = ptweets['Tweets']
                    if(len(ptweets)>0):
                        st.subheader('Positive tweets and percentage of positive tweets')
                        st.write(ptweets)
                        st.markdown('##')
                        pos_per = round((ptweets.shape[0] / df.shape[0]) * 100, 1)
                        st.write("{} tweet is:{}% positive".format(user.name, pos_per))
                    else:
                        st.header("No positive tweets from user: {}".format(user.name))
                    st.markdown('##')
                    st.markdown('##')
                    # get the percentage of negative tweets
                    ntweets = df[df.Analysis == 'Negative']
                    ntweets = ntweets['Tweets']
                    if(len(ntweets)>0):
                        st.subheader('negative tweets and percentage of negative tweets')
                        st.write(ntweets)
                        st.markdown('##')
                        neg_per = round((ntweets.shape[0] / df.shape[0]) * 100, 1)
                        st.write("{} tweet is:{}% negative".format(user.name, neg_per))
                    else:
                        st.header("No Negative  tweets from user: {}".format(user.name))

                    st.markdown('##')
                    st.markdown('##')
                    try:
                        fig1 = plt.figure(figsize=(8, 6))
                        count_pos = len(ptweets)
                        count_neg = len(ntweets)
                        count_neu = len(df) - len(ptweets) - len(ntweets)
                        data = [count_pos, count_neg, count_neu - 1]
                        keys = ['Positive', 'Negative', 'Neutral']
                        explode = [0, 0.1, 0]
                        palette_color = sns.color_palette('dark')
                        plt.pie(data, labels=keys, colors=palette_color, explode=explode, autopct='%.0f%%')
                        st.pyplot(fig1)
                    except Exception as e:
                        st.warning("Error occured:{}".format(e))
                    st.markdown('##')
                    st.markdown('##')






    elif choice=='query':
        # tweet query
        ### TWEET SEARCH AND CLASSIFY ###
        st.subheader('Search Twitter for Query')
        with st.form(key='query form',clear_on_submit=True):
            # Get user input
            keyword = st.text_input('Query:', '#')
            count_number = st.number_input("Enter number of tweets you want to query", min_value=10,
                                           max_value=1500,
                                           step=5)
            submit_text1 = st.form_submit_button(label='Submit')

        def tweets_query(api, query, count):
            tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en").items(count)
            # tweets
            tweets_list = [
                [tweet.text, tweet.id, tweet.source, tweet.coordinates, tweet.retweet_count,
                 tweet.favorite_count,tweet.created_at,
                 tweet.user._json['name'], tweet.user._json['screen_name'], tweet.user._json['location'],
                 tweet.user._json['friends_count'],
                 tweet.user._json['verified'], tweet.user._json['description'],
                 tweet.user._json['followers_count']] for
                tweet in tweets]
            tweets_df = pd.DataFrame(tweets_list,
                                     columns=['tweet_text', 'tweet_id', 'tweet_source', 'coordinates',
                                              'retweet_count',
                                              'likes_count','created date', 'Username', 'screen_name', 'location',
                                              'friends_count',
                                              'verification_status', 'description', 'followers_count'])
            return tweets_df

        query = keyword
        count = count_number
        if submit_text1:
            with st.spinner(f'searching & analyzing {query} for {count } count.....'):
                sleep(10)
                tweets = tweets_query(api, query, count)
                tweetdf = (tweets
                    .sort_values(['followers_count'], ascending=False)
                [['tweet_text', 'retweet_count', 'followers_count','created date']])
                tweetdf['tweet_text'] = tweetdf['tweet_text'].astype(str)
                print(len(tweetdf))
                if(len(tweetdf)==0):
                    st.warning("There are no tweet related to this query, please input another query")
                    ph = st.empty()
                    N = 10
                    bar = st.progress(0)
                    for secs in range(0, N, 1):
                        mm, ss = (N - secs) // 60, (N - secs) % 60
                        bar.progress((secs + 1) * 10)
                        ph.metric("Redirecting in...", f"{mm:02d}:{ss:02d}")
                        sleep(1)

                    st.experimental_rerun()
                else:
                    def cleanTxt(text):
                        text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
                        text = re.sub('#', '', text)  # Removing '#' hash tag
                        text = re.sub('RT[\s]+', '', text)  # Removing RT
                        text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink

                        return text

                    def getSubjectivity(text):
                        return TextBlob(text).sentiment.subjectivity

                    # Create a function to get the polarity
                    def getPolarity(text):
                        return TextBlob(text).sentiment.polarity

                    def getAnalysis(score):
                        if score < 0:
                            return 'Negative'
                        elif score == 0:
                            return 'Neutral'
                        else:
                            return 'Positive'

                    tweetdf['tweet_text'] = tweetdf['tweet_text'].apply(cleanTxt)
                    tweetdf['Polarity'] = tweetdf['tweet_text'].apply(getPolarity)
                    tweetdf['Analysis'] = tweetdf['Polarity'].apply(getAnalysis)
                    tweetdf['Sentiment'] = tweetdf['tweet_text'].apply(predict_emotions)
                    tweetdf['Subjectivity'] = tweetdf['tweet_text'].apply(getSubjectivity)
                    st.write(tweetdf)
                    # plot the polarity and subjectivity
                    st.markdown('##')
                    st.markdown('##')
                    st.subheader("Polarity VS subjectivity graph")
                    fig = plt.figure(figsize=(8, 6))
                    for i in range(0, tweetdf.shape[0]):
                        plt.scatter(tweetdf['Polarity'][i], tweetdf['Subjectivity'][i], color='Blue')
                    plt.title('Sentiment Analysis')
                    plt.xlabel('Polarlity')
                    plt.ylabel('Subjectivity')
                    st.pyplot(fig)
                    st.markdown('##')
                    st.markdown('##')
                    # get the percentage of positive tweets
                    st.subheader('Positive tweets and percentage of positive tweets')
                    ptweets = tweetdf[tweetdf.Analysis == 'Positive']
                    ptweets = ptweets['tweet_text']
                    st.write(ptweets)
                    st.markdown('##')
                    pos_per = round((ptweets.shape[0] / tweetdf.shape[0]) * 100, 1)
                    st.write("{} query tweet is:{}% positive".format(query, pos_per))
                    st.markdown('##')
                    st.markdown('##')
                    # get the percentage of negative tweets
                    st.subheader('negative tweets and percentage of negative tweets')
                    ntweets = tweetdf[tweetdf.Analysis == 'Negative']
                    ntweets = ntweets['tweet_text']
                    st.write(ntweets)
                    st.markdown('##')
                    neg_per = round((ntweets.shape[0] / tweetdf.shape[0]) * 100, 1)
                    st.write("{} query  tweet is:{}% negative".format(query, neg_per))
                    st.markdown('##')
                    st.markdown('##')
                    try:
                        fig1 = plt.figure(figsize=(8, 6))
                        count_pos = len(ptweets)
                        count_neg = len(ntweets)
                        count_neu = len(tweetdf) - len(ptweets) - len(ntweets)
                        data = [count_pos, count_neg, count_neu - 1]
                        keys = ['Positive', 'Negative', 'Neutral']
                        explode = [0, 0.1, 0]
                        palette_color = sns.color_palette('dark')
                        plt.pie(data, labels=keys, colors=palette_color, explode=explode, autopct='%.0f%%')
                        st.pyplot(fig1)
                    except Exception as e:
                        st.warning("Error occured:{}".format(e))

                    st.markdown('##')
                    st.markdown('##')


    else:
        about()

if __name__ == '__main__':
    main()
