def generateWordCloud(text_data, title):
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(text_data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

    
    #WORD CLOUD
    all_titles = [post[1] for post in trends['Body']]
    # generateWordCloud(all_titles, 'Word Cloud of Post Titles')
    plt.tight_layout()
    plt.show()