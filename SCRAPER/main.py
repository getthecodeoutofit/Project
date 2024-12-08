from Scripts import RedditScraper
import os
from dotenv import load_dotenv
load_dotenv()
import WordCloud

if __name__ == "__main__":
    scraper = RedditScraper(
        client_id=os.getenv('C_ID'),
        client_secret=os.getenv('C_SECRET'),
        user_agent=os.getenv('USER_AGENT'),
        db_host=os.getenv('D_HOST'),
        db_user=os.getenv('D_USER'),
        db_password=os.getenv('PASS'),
        db_name=os.getenv('D_NAME')
    )
    

    while(True):

        print('''1 - Create Tables:\t\t\t2 - Post Scraping:\t\t\t3 - Sraper data By USER ID:\n
4 - Export To CSV:\t\t\t5 - Plot Graph/Analysis:\t\t\t6 - Clear DATABASE:\n
7 - Search By PostID:\t\t\t8 - Predict UPVOTES\t\t\t9 - Generate Word cloud:
        ''')
        choice = int(input("ENTER YOUR CHOICE: "))

        match(choice):
            case 1:
                scraper.createTables()

            case 2:
                limit = int(input("Enter your Limit: "))
                topic = input("Enter the subreddit/topic: ")
                scraper.scrape_posts(topic, limit)

            case 3:
                usr = input("Enter the user ID: ")
                result = scraper.searchById(usr)
                print("Search result:", result)

            case 4:
                posts_csv, comments_csv = scraper.toCSV()
                print(f"Data exported to {posts_csv} and {comments_csv}")

            case 5:
                scraper.plot_advanced_analytics()

            case 6:
                print("CLEARING DATABASE: ")
                scraper.clearDB()
            case 7:
                postid = input("Enter the ID: ")
                scraper.searchByPost(postid)

            
            case 8:
                postid = input("Enter the Post ID: ")
                prediction_result = scraper.predictPostPerformance(postid)
                if "error" in prediction_result:
                    print("Error:", prediction_result["error"])
                else:
                    print("Prediction Results:")
                    print(f"Post ID: {prediction_result['post_id']}")
                    print(f"Predicted Upvotes: {prediction_result['predicted_upvotes']}")
                    print("Model Metrics:")
                    print(f"Mean Squared Error: {prediction_result['model_metrics']['mean_squared_error']:.2f}")
                    print(f"R2 Score: {prediction_result['model_metrics']['r2_score']:.2f}")
            
            case 9:
                sub = input("Enter the name of Subreddit:")
                WordCloud.gen(sub)
            
            case 10:
                break

            case _:
                print("please select a valid Choice number: ")