import logging.config
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def generate_word_cloud(db_cursor, table='Posts', text_column='Title', subreddit=None):
    try:
        # Construct query
        query = f"SELECT {text_column} FROM {table}"
        if subreddit:
            query += " WHERE Subreddit = %s"
            db_cursor.execute(query, (subreddit,))
        else:
            db_cursor.execute(query)

        # Fetch all text data
        text_data = db_cursor.fetchall()
        combined_text = " ".join([row[0] for row in text_data if row[0]])

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

        # Plot word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Word Cloud")
        plt.show()
    except Exception as e:
        logger.error(f"Error generating word cloud: {e}")

db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '1212',
            'database': 'RDnew'
        }
import pymysql

con = pymysql.connect(**db_config)
db = con.cursor()

def gen(sub):
    generate_word_cloud(db,subreddit=sub)