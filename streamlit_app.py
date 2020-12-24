import streamlit as st
import joblib
import os

DIRNAME = os.path.dirname(__file__)

CLASS_LIST = ['action', 'adventure', 'animation', 'biography', 'comedy',
              'crime', 'drama', 'family', 'fantasy', 'history', 'horror',
              'music', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller',
              'war']


def get_pkl(filepath):
    filename = os.path.join(DIRNAME, filepath)
    return joblib.load(filename)


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
