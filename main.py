import joblib

vectorizer = joblib.load("news_vectorizer_dump.pkl")
our_model = joblib.load("news_model_dump.pkl")

class_list = ['action', 'adventure', 'animation', 'biography', 'comedy',
              'crime', 'drama', 'family', 'fantasy', 'history', 'horror',
              'music', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller',
              'war']

our_text = input()

class_index = our_model.predict(vectorizer.transform(our_text))[0]
print(class_list[class_index])