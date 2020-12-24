import joblib
import os
import re
import streamlit as st
from nltk.stem import WordNetLemmatizer

DIRNAME = os.path.dirname(__file__)

CLASS_LIST = ['action', 'adventure', 'animation', 'biography', 'comedy',
              'crime', 'drama', 'family', 'fantasy', 'history', 'horror',
              'music', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller',
              'war']


def get_pkl(filepath):
    filename = os.path.join(DIRNAME, filepath)
    return joblib.load(filename)

def tranform_our_string(our_string):
    stemmer = WordNetLemmatizer()

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
    our_string = document.split()

    our_string = [stemmer.lemmatize(word) for word in our_string]
    our_string = ' '.join(our_string)
    



def main():
    vectorizer = get_pkl('news_vectorizer_dump.pkl')
    our_model = get_pkl('news_model_dump.pkl')
    # our_text = [input()]
    st.title('Last ML home_work')
    # our_text = st.text('Input your text')
    # with st.echo():
    our_text = st.text_input("Input text")
    class_index = our_model.predict(vectorizer.transform([our_text]))[0]
    st.text(CLASS_LIST[class_index])
    # our_text += 'ADD'
    # st.write(our_text.text)
    # print(CLASS_LIST[class_index])


if __name__ == "__main__":
    main()
