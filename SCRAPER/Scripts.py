import praw
import pymysql
import datetime
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Optional, List, Dict, Union
import logging
import os
from datetime import timedelta
from dotenv import load_dotenv
from wordcloud import WordCloud
import time
from sklearn.model_selection import train_test_split

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#------------------------------------------------


class RedditScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str,
                 db_host: str, db_user: str, db_password: str, db_name: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.db_config = {
            'host': db_host,
            'user': db_user,
            'password': db_password,
            'database': db_name
        }
        self.connect_db()

    def connect_db(self):
        try:
            self.db_connection = pymysql.connect(**self.db_config)
            self.db_cursor = self.db_connection.cursor()
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def createTables(self):
        """Create necessary database tables"""
        try:
            # Posts table with additional metadata
            self.db_cursor.execute("""
                CREATE TABLE IF NOT EXISTS Posts (
                    Id VARCHAR(255) PRIMARY KEY,
                    Title TEXT,
                    Subreddit VARCHAR(255),
                    Author VARCHAR(255),
                    Upvotes INT,
                    Created_At DATETIME,
                    Sentiment FLOAT,
                    Url TEXT,
                    Is_Original_Content BOOLEAN,
                    Word_Count INT,
                    Last_Updated DATETIME
                );
            """)

            self.db_cursor.execute("""
                CREATE TABLE IF NOT EXISTS Comments (
                    Id VARCHAR(255) PRIMARY KEY,
                    Post_Id VARCHAR(255),
                    Author VARCHAR(255),
                    Body TEXT,
                    Upvotes INT,
                    Created_At DATETIME,
                    Sentiment FLOAT,
                    Parent_Id VARCHAR(255),
                    Is_Top_Level BOOLEAN,
                    Word_Count INT,
                    FOREIGN KEY (Post_Id) REFERENCES Posts(Id) ON DELETE CASCADE
                );
            """)

            self.db_cursor.execute("""
                CREATE TABLE IF NOT EXISTS SubredditStats (
                    Subreddit VARCHAR(255) PRIMARY KEY,
                    Last_Scraped DATETIME,
                    Total_Posts INT,
                    Average_Sentiment FLOAT,
                    Average_Upvotes FLOAT
                );
                                   """)
                                   
            self.db_cursor.execute("""
                CREATE TABLE IF NOT EXISTS Particular_Data (
                    Id INT AUTO_INCREMENT PRIMARY KEY,
                    Post_Id VARCHAR(255),
                    User_Id VARCHAR(255),
                    User_Number INT,
                    Post_Upvotes int,
                    Parsed varchar(255),
                    FOREIGN KEY (Post_Id) REFERENCES Posts(Id)
                );
            """)
            
            self.db_connection.commit()
            logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            self.db_connection.rollback()

    def scrape_posts(self, subreddit_name: str, limit: int = 5, time_filter: str = 'week'):
        """Scrape posts with enhanced metadata"""
        subreddit = self.reddit.subreddit(subreddit_name) 

        for post in subreddit.top(time_filter=time_filter, limit=limit):
            sentiment = TextBlob(post.title).sentiment.polarity
            word_count = len(post.title.split())
            
            try:
                self.db_cursor.execute("""
                    INSERT IGNORE INTO Posts 
                    (Id, Title, Subreddit, Author, Upvotes, Created_At, Sentiment, 
                     Url, Is_Original_Content, Word_Count, Last_Updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    post.id, post.title, subreddit_name, str(post.author),
                    post.score, datetime.datetime.fromtimestamp(post.created_utc),
                    sentiment, post.url, post.is_original_content, word_count,
                    datetime.datetime.now()
                ))
                self.db_connection.commit()
                self.scrapeComments(post)
                logger.info(f"Successfully scraped post: {post.id}")
            except Exception as e:
                logger.error(f"Error inserting post {post.id}: {e}")
                self.db_connection.rollback()

    def updateSubredditStats(self, subreddit_name: str):
        try:
            # Calculate aggregate statistics for the subreddit
            self.db_cursor.execute("""
                SELECT 
                    COUNT(*) as total_posts,
                    AVG(Sentiment) as average_sentiment,
                    AVG(Upvotes) as average_upvotes
                FROM Posts
                WHERE Subreddit = %s
            """, (subreddit_name,))
            stats = self.db_cursor.fetchone()

                # Update or insert into the SubredditStats table
            self.db_cursor.execute("""
                    INSERT INTO SubredditStats 
                    (Subreddit, Last_Scraped, Total_Posts, Average_Sentiment, Average_Upvotes)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    Last_Scraped = VALUES(Last_Scraped),
                    Total_Posts = VALUES(Total_Posts),
                    Average_Sentiment = VALUES(Average_Sentiment),
                    Average_Upvotes = VALUES(Average_Upvotes)
            """, (subreddit_name, datetime.datetime.now(), stats[0], stats[1], stats[2]))
            self.db_connection.commit()
            logger.info(f"Updated SubredditStats for {subreddit_name}")
        except Exception as e:
            logger.error(f"Error updating SubredditStats for {subreddit_name}: {e}")
            self.db_connection.rollback()


    def scrapeComments(self, post):
        """Scrape comments with enhanced metadata"""
        post.comments.replace_more(limit=0)
        for comment in post.comments.list():
            sentiment = TextBlob(comment.body).sentiment.polarity
            word_count = len(comment.body.split())
            
            try:
                self.db_cursor.execute("""
                    INSERT IGNORE INTO Comments 
                    (Id, Post_Id, Author, Body, Upvotes, Created_At, Sentiment,
                     Parent_Id, Is_Top_Level, Word_Count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    comment.id, post.id, str(comment.author), comment.body,
                    comment.score, datetime.datetime.fromtimestamp(comment.created_utc),
                    sentiment, comment.parent_id, comment.parent_id == post.id,
                    word_count
                ))
                self.db_connection.commit()
            except Exception as e:
                logger.error(f"Error inserting comment {comment.id}: {e}")
                self.db_connection.rollback()

    def scrape_posts_by_date(self, subreddit_name: str, from_date: str, to_date: str):
        try:
            start_timestamp = int(time.mktime(datetime.datetime.strptime(from_date, "%Y-%m-%d").timetuple()))
            end_timestamp = int(time.mktime(datetime.datetime.strptime(to_date, "%Y-%m-%d").timetuple()))
            
            subreddit = self.reddit.subreddit(subreddit_name)
            query = f'timestamp:{start_timestamp}..{end_timestamp}'
            
            for post in subreddit.search(query, sort="new", syntax="cloudsearch"):
                sentiment = TextBlob(post.title).sentiment.polarity
                word_count = len(post.title.split())
                
                try:
                    self.db_cursor.execute("""
                        INSERT IGNORE INTO Posts 
                        (Id, Title, Subreddit, Author, Upvotes, Created_At, Sentiment, 
                         Url, Is_Original_Content, Word_Count, Last_Updated)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        post.id, post.title, subreddit_name, str(post.author),
                        post.score, datetime.datetime.fromtimestamp(post.created_utc),
                        sentiment, post.url, post.is_original_content, word_count,
                        datetime.datetime.now()
                    ))
                    self.db_connection.commit()
                    self.scrapeComments(post)
                    logger.info(f"Successfully scraped post: {post.id}")
                except Exception as e:
                    logger.error(f"Error inserting post {post.id}: {e}")
                    self.db_connection.rollback()

        except Exception as e:
            logger.error(f"Error scraping posts for date range {from_date} to {to_date}: {e}")



    def searchById(self, search_id: str) -> Dict:

        result = {'post': None, 'comments': []}
        self.db_cursor.execute("""SELECT * FROM Posts WHERE Id = %s""", (search_id,))
        post = self.db_cursor.fetchone()
        
        if post:
            result['post'] = {
                'id': post[0],
                'title': post[1],
                'subreddit': post[2],
                'author': post[3],
                'upvotes': post[4],
                'created_at': post[5],
                'sentiment': post[6]
            }

            self.db_cursor.execute("""SELECT * FROM Comments WHERE Post_Id = %s ORDER BY Created_At DESC""", (search_id,))
            comments = self.db_cursor.fetchall()
            result['comments'] = [{
                'id': comment[0],
                'author': comment[2],
                'body': comment[3],
                'upvotes': comment[4],
                'created_at': comment[5],
                'sentiment': comment[6]
            } for comment in comments]
        else:
            self.db_cursor.execute("""SELECT * FROM Comments WHERE Id = %s""", (search_id,))
            comment = self.db_cursor.fetchone()
            if comment:
                result['comment'] = {
                    'id': comment[0],
                    'post_id': comment[1],
                    'author': comment[2],
                    'body': comment[3],
                    'upvotes': comment[4],
                    'created_at': comment[5],
                    'sentiment': comment[6]
                }
        
        return result

    def toCSV(self, output_dir: str = 'exports'):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            self.db_cursor.execute("SELECT * FROM Posts")
            posts = self.db_cursor.fetchall()
            posts_df = pd.DataFrame(posts, columns=[
                'Id', 'Title', 'Subreddit', 'Author', 'Upvotes', 'Created_At',
                'Sentiment', 'Url', 'Is_Original_Content', 'Word_Count', 'Last_Updated'
            ])
            posts_df.to_csv(f"{output_dir}/posts_{timestamp}.csv", index=False)
            
            self.db_cursor.execute("SELECT * FROM Comments")
            comments = self.db_cursor.fetchall()
            comments_df = pd.DataFrame(comments, columns=[
                'Id', 'Post_Id', 'Author', 'Body', 'Upvotes', 'Created_At',
                'Sentiment', 'Parent_Id', 'Is_Top_Level', 'Word_Count'
            ])
            comments_df.to_csv(f"{output_dir}/comments_{timestamp}.csv", index=False)
            
            logger.info(f"Data exported successfully to {output_dir}")
            return f"{output_dir}/posts_{timestamp}.csv", f"{output_dir}/comments_{timestamp}.csv"
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise

    

    def analyseSentimentTrends(self, subreddit: Optional[str] = None, 
                               days: int = 30) -> Dict:
        query = """
            SELECT 
                DATE(Created_At) as date,
                AVG(Sentiment) as avg_sentiment,
                COUNT(*) as post_count,
                AVG(Upvotes) as avg_upvotes
            FROM Posts
            WHERE Created_At >= DATE_SUB(NOW(), INTERVAL %s DAY)
        """
        if subreddit:
            query += " AND Subreddit = %s"
            self.db_cursor.execute(query + " GROUP BY DATE(Created_At)", (days, subreddit))
        else:
            self.db_cursor.execute(query + " GROUP BY DATE(Created_At)", (days,))
        
        results = self.db_cursor.fetchall()
        self.db_cursor.execute('Select Title from Posts')
        resP = self.db_cursor.fetchall()
        return {
            'dates': [row[0] for row in results],
            'sentiment': [row[1] for row in results],
            'post_count': [row[2] for row in results],
            'avg_upvotes': [row[3] for row in results],
            'Body':[row[0] for row in resP]
        }
    
    def parseCommentsWithPost(self,postId:str):
        print('Parsing The Comments of the particular post id: ')
        try:
            self.db_cursor.execute(f'''SELECT * FROM c JOINS Posts P on Comments c where PostId = {postId} ''',(postId))
            data = self.db_cursor.fetchall()
            print(data)

        except Exception as e:
            logger.error("Error In Parsing the Comments: ")
            raise

    def printPost(self):
        pass
        try:
            self.db_cursor

        except Exception as e:
            logger.error("Enter in Fetching/Printing Posts From the database: ")
            raise


    
    def plot_advanced_analytics(self):
        trends = self.analyseSentimentTrends() 
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sentiment over time
        ax1.plot(trends['dates'], trends['sentiment'], marker='o')
        ax1.set_title('Sentiment Trends')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Average Sentiment')
        
        # Post volume
        ax2.bar(trends['dates'], trends['post_count'])
        ax2.set_title('Post Volume')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Posts')
        
        ax3.scatter(trends['sentiment'], trends['avg_upvotes'])
        ax3.set_title('Upvotes vs Sentiment')
        ax3.set_xlabel('Sentiment')
        ax3.set_ylabel('Average Upvotes')
        
        # Engagement heat map
        hour_data = self.getHourlyEngagement()
        im = ax4.imshow(hour_data)
        ax4.set_title('Engagement by Hour')
        plt.colorbar(im, ax=ax4)
        plt.tight_layout()
        plt.show()

    def searchByPost(self,post_id):
        post  = self.reddit.submission(id=post_id)
        user_id = str(post.author)
        user_number = hash(user_id) % 1000000            
        try:
            self.db_cursor.execute("""
                    INSERT IGNORE INTO Particular_Data 
                    (Post_Id, User_Id, User_Number, Post_Upvotes, Parsed)
                    VALUES (%s, %s, %s)
                """, (
                    post.id, user_id, user_number, post.upvote, post.parse
                ))     
            self.db_connection.commit()
            logger.info(f"Post {post_id} scraped and inserted successfully")
        except Exception as e:
            logger.error(f"Error scraping post {post_id}: {e}")
            raise
        

    def getHourlyEngagement(self) -> np.ndarray:
        """Get engagement metrics by hour"""
        self.db_cursor.execute("""
            SELECT HOUR(Created_At) as hour,
                   DAYOFWEEK(Created_At) as day,
                   AVG(Upvotes) as avg_engagement
            FROM Posts
            GROUP BY HOUR(Created_At), DAYOFWEEK(Created_At)
        """)
        results = self.db_cursor.fetchall()
        
        engagement_matrix = np.zeros((24, 7))
        for hour, day, engagement in results:
            engagement_matrix[hour, day-1] = engagement
        return engagement_matrix


    def predictPostPerformance(self, post_id: str):
        try:
            # Fetch historical data
            self.db_cursor.execute("""
                SELECT Upvotes, Sentiment, Word_Count, Is_Original_Content
                FROM Posts
                WHERE Id != %s
            """, (post_id,))
            data = self.db_cursor.fetchall()

            if not data:
                logger.error("Not enough historical data for training")
                return {"error": "Not enough historical data for training"}

            # Convert data to a DataFrame
            columns = ['Upvotes', 'Sentiment', 'Word_Count', 'Is_Original_Content']
            df = pd.DataFrame(data, columns=columns)

            # Prepare features (X) and target (y)
            X = df[['Sentiment', 'Word_Count', 'Is_Original_Content']]
            y = df['Upvotes']

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Evaluate model performance
            y_pred = model.predict(X_test)
            from sklearn.metrics import root_mean_squared_error,r2_score
            mse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Fetch the input post's data
            self.db_cursor.execute("""
                SELECT Sentiment, Word_Count, Is_Original_Content
                FROM Posts
                WHERE Id = %s
            """, (post_id,))
            post_data = self.db_cursor.fetchone()

            if not post_data:
                logger.error(f"No data found for post_id: {post_id}")
                return {"error": "Post data not found"}

            # Predict the upvotes for the given post
            post_features = np.array(post_data).reshape(1, -1)
            predicted_upvotes = model.predict(post_features)[0]

            # Return the prediction and metrics
            result = {
                "post_id": post_id,
                "predicted_upvotes": round(predicted_upvotes),
                "model_metrics": {
                    "mean_squared_error": mse,
                "r2_score": r2
            }
            }

            logger.info(f"Prediction for post {post_id}: {result}")
            return result

        except Exception as e:
            logger.error(f"Error predicting post performance: {e}")
            return {"error": str(e)}
        
    


    def clearDB(self) -> bool:
        try:
            self.db_cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            self.db_cursor.execute("TRUNCATE TABLE Comments")
            self.db_cursor.execute("TRUNCATE TABLE Posts")
            self.db_cursor.execute("TRUNCATE TABLE SubredditStats")
            self.db_cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            self.db_connection.commit()
            logger.info("Database cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            self.db_connection.rollback()
            return False
        
    
    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db_connection') and self.db_connection:
            self.db_connection.close()





