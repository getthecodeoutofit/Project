import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

# Fetch posts from MySQL
def fetch_post_text(cursor, subreddit=None, from_date=None, to_date=None):
    query = "SELECT Title, SelfText FROM Posts WHERE 1=1"
    if subreddit:
        query += f" AND Subreddit = '{subreddit}'"
    if from_date and to_date:
        query += f" AND Created_At BETWEEN '{from_date}' AND '{to_date}'"
    
    cursor.execute(query)
    rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=['Title', 'SelfText'])

def train_lda(posts_df, num_topics=5):
    posts_df['Processed_Text'] = posts_df['Title'].fillna('') + ' ' + posts_df['SelfText'].fillna('')
    posts_df['Processed_Text'] = posts_df['Processed_Text'].apply(preprocess_text)

    # Create dictionary and corpus
    dictionary = Dictionary(posts_df['Processed_Text'])
    corpus = [dictionary.doc2bow(text) for text in posts_df['Processed_Text']]

    # Train LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)
    return lda_model, corpus, dictionary

# Visualize topics using pyLDAvis
def visualize_topics(lda_model, corpus, dictionary):
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    return vis

# Function to generate and visualize topics
def generate_topics(cursor, subreddit=None, from_date=None, to_date=None, num_topics=5):
    # Fetch posts
    posts_df = fetch_post_text(cursor, subreddit, from_date, to_date)
    if posts_df.empty:
        return "No posts found for the given criteria."

    # Train and visualize topics
    lda_model, corpus, dictionary = train_lda(posts_df, num_topics=num_topics)
    vis = visualize_topics(lda_model, corpus, dictionary)
    return lda_model.print_topics(num_words=10), vis
