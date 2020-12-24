import joblib
import os
import re
import streamlit as st
import nltk
nltk.download('wordnet')

DIRNAME = os.path.dirname(__file__)

CLASS_LIST = ['action', 'adventure', 'animation', 'biography', 'comedy',
              'crime', 'drama', 'family', 'fantasy', 'history', 'horror',
              'music', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller',
              'war']

AGGRESSIVE_GENRE = ['action', 'crime', 'horror', 'thriller', 'war']


def get_pkl(filepath):
    filename = os.path.join(DIRNAME, filepath)
    return joblib.load(filename)


def tranform_our_string(our_string):
    stemmer = nltk.stem.WordNetLemmatizer()
    # Remove all the special characters
    our_string = re.sub(r'\W', ' ', our_string)
    # remove all single characters
    our_string = re.sub(r'\s+[a-zA-Z]\s+', ' ', our_string)
    # Remove single characters from the start
    our_string = re.sub(r'\^[a-zA-Z]\s+', ' ', our_string)
    # Substituting multiple spaces with single space
    our_string = re.sub(r'\s+', ' ', our_string, flags=re.I)
    # Removing prefixed 'b'
    our_string = re.sub(r'^b\s+', '', our_string)
    # Converting to Lowercase
    our_string = our_string.lower()
    # Lemmatization
    our_string = our_string.split()
    our_string = [stemmer.lemmatize(word) for word in our_string]
    our_string = ' '.join(our_string)
    return our_string


def main():
    vectorizer = get_pkl('news_vectorizer_dump.pkl')
    our_model = get_pkl('news_model_dump.pkl')
    st.title('Последнее домашнее задание по МЛ')
    our_text = st.text_input("Введите текст на английском языке")
    our_text = tranform_our_string(our_text)
    class_index = our_model.predict(vectorizer.transform([our_text]))[0]
    answer = CLASS_LIST[class_index]
    st.text(f"Genre: {answer}")
    if answer in AGGRESSIVE_GENRE:
        st.text("ALARM!!!")
    else:
        st.text("Normal situation")

if __name__ == "__main__":
    main()
