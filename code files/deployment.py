import pandas as pd
import streamlit as st
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
import time

# Load required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the vectorizer and sentiment analysis model
try:
    vectorizer = pickle.load(open("C:\\Users\\Home\\Desktop\\comments\\vector.pkl", 'rb'))
    model = pickle.load(open("C:\\Users\\Home\\Desktop\\comments\\model.pkl", 'rb'))
except Exception as e:
    st.error(f"Error loading vectorizer or model: {str(e)}")
    st.stop()

# Function to remove punctuation
def remove_punctuation(text):
    cleaned_text = re.sub(r"[^a-zA-Z#]", " ", text)
    return cleaned_text

# Function to lemmatize and join text
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lemmatize_and_join(text):
    words = text.split()
    lemmatized_words = [wnl.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(lemmatized_words)

# Streamlit configuration
st.set_page_config(page_title='YouTube Sentiment Analysis', page_icon="▶️")

# Load the uploaded dataset
comments_column = 'text'  # Replace this with the actual column name if different
# Load the uploaded dataset
file_path = 'C:\\Users\\Home\\Desktop\\comments\\sentiment_dataset_with_sentiment_processed.csv'
try:
    comments_df = pd.read_csv(file_path)
    #st.write("DataFrame columns:", comments_df.columns)  # Display columns to identify the correct one

    # Convert comments to string type if not already
    comments_df[comments_column] = comments_df[comments_column].astype(str)
except Exception as e:
    st.error(f"Error loading CSV file: {str(e)}")
    st.stop()


# Identify the correct column name for comments


# Streamlit layout and input
colIm, coltit = st.columns([1, 3])
colIm.image("C:\\Users\\Home\\Desktop\\comments\\youtube-logo.png", width=160)
coltit.title("YouTube Video Sentiments Analysis")

st.sidebar.title("About this webApp")
st.sidebar.write("The YouTube Video Sentiments Analysis web app allows users to analyze the sentiment distribution of comments on a given YouTube video. By entering the video link and selecting the percentage of comments to analyze, it performs sentiment analysis to categorize comments as positive, negative, or neutral. The app performs sentiment analysis, engagement analysis, and offers time analysis for viewer engagement. It provides valuable insights for content creators and marketers.")
st.sidebar.write("-- BUKC")
st.sidebar.markdown("<p style='color: red;'>Please note that this web app works on maximum accuracy and is a part of projects so may only be available for a few users.</p>", unsafe_allow_html=True)

# Text input for video link
st.markdown("Credits: Dawood & Osama")

# Slider to select the percentage of comments to retrieve
selected_percentage = st.slider('Percentage of comments to analyze:', 20, 70, step=10)

# Calculate the number of comments based on selected percentage
total_comments = len(comments_df)
desired_comments = int((selected_percentage / 100) * total_comments)

# Limit the number of comments to retrieve based on the desired_comments count
if desired_comments > 0:
    try:
        with st.spinner(f"Loading {desired_comments} comments..."):
            # Limit the DataFrame to the desired number of comments
            comments_df = comments_df.head(desired_comments)

        # Preprocess comments
        comments_df['Processed Comment'] = comments_df[comments_column].apply(remove_punctuation)
        comments_df['Processed Comment'] = comments_df['Processed Comment'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
        comments_df['Processed Comment'] = comments_df['Processed Comment'].apply(lambda x: x.lower())
        comments_df['Processed Comment'] = comments_df['Processed Comment'].apply(lemmatize_and_join)

        # Perform sentiment analysis
        comments_df['Transformed Comment'] = comments_df['Processed Comment'].apply(lambda x: vectorizer.transform([x]).toarray()[0])
        comments_df['Sentiment'] = comments_df['Transformed Comment'].apply(lambda x: model.predict([x])[0])

        # Define sentiment mapping
        sentiment_mapping = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}
        comments_df['Sentiment Label'] = comments_df['Sentiment'].map(sentiment_mapping)
        sentiment_counts = comments_df['Sentiment Label'].value_counts()

        # Visualization
        col1, col2 = st.columns(2)

        total_comments = sentiment_counts.sum()
        percentage_positive = (sentiment_counts['Positive'] / total_comments) * 100 if 'Positive' in sentiment_counts else 0
        percentage_neutral = (sentiment_counts['Neutral'] / total_comments) * 100 if 'Neutral' in sentiment_counts else 0
        percentage_negative = 100 - percentage_positive - percentage_neutral
        col1.write('Sentiment Distribution')
        plt.style.use('dark_background')

        # Plot the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [percentage_positive, percentage_neutral, percentage_negative]
        colors = ['#00adb5', '#ff6b6b', '#ffd166']
        explode = (0.1, 0, 0)
        plt.figure(figsize=(8, 5))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, textprops={'color': 'black'})
        plt.axis('equal')
        plt.rcParams['text.color'] = 'white'
        for wedge in plt.gca().patches:
            wedge.set_edgecolor('black')
        plt.title('Sentiment Distribution of Comments', color='white', fontsize=16)
        plt.legend(labels, loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=12)
        col1.pyplot(plt)
        # Display the most liked comment
        most_liked_comment = comments_df.iloc[comments_df['Likes'].idxmax()]['Processed Comment'] if 'Likes' in comments_df.columns and not comments_df.empty else "No comments"
        most_likes = comments_df['Likes'].max() if 'Likes' in comments_df.columns else 0
        col2.write(f"Most Liked Comment ({most_likes} likes):")
        col2.write(f'"->{most_liked_comment}')

        # Engagement analysis
        if 'Likes' in comments_df.columns and 'Sentiment Label' in comments_df.columns:
            total_video_likes = comments_df['Likes'].sum()
            comments_df['Engagement Rate'] = (comments_df['Likes'] / total_video_likes) * 100 if total_video_likes > 0 else 0
            average_engagement_by_sentiment = comments_df.groupby('Sentiment Label')['Engagement Rate'].mean()

            # Plot the bar chart to visualize the relationship between sentiment and average engagement rate
            plt.figure(figsize=(8, 5))
            average_engagement_by_sentiment.plot(kind='bar', color=['#66bb6a', '#ffca28', '#ef5350'])
            plt.xticks(rotation=0)
            plt.xlabel('Sentiment')
            plt.ylabel('Average Engagement Rate (%)')
            plt.title('Relationship between Sentiment and Engagement')
            plt.grid(axis='y')
            plt.tight_layout()
            st.pyplot(plt)
        else:
            col2.write("No likes data available for engagement analysis.")

        # Word cloud function
        def create_word_cloud(sentiment_category):
            comments_text = ' '.join(comments_df[comments_df['Sentiment Label'] == sentiment_category]['Processed Comment'])
            if not comments_text:
                st.write(f"No {sentiment_category} comments found.")
                return
            wordcloud = WordCloud(width=800, height=400, background_color='black').generate(comments_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for {sentiment_category} Comments')
            st.pyplot(plt)

        # Create word clouds for positive, negative, and neutral comments
        create_word_cloud('Positive')
        create_word_cloud('Negative')
        create_word_cloud('Neutral')
    except Exception as e:
        st.error(f"Failed to process comments: {str(e)}")
else:
    st.write("No comments available for the selected percentage.")

