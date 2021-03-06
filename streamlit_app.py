"""Simple text classification application."""
import os
import random
import re

import joblib
import nltk
import streamlit as st

nltk.download('wordnet')

DIRNAME = os.path.dirname(__file__)
TEXT_FROM_CURRENT_PERSON = "Text from current person:"
EXAMPLE_TEXT = [
    "I love everyone",
    "I want to kill everyone",
    "Only the stupid ones",
    "Do you remember who gave us this",
    "Jesus Pop how can you stand the cold dressed like that",
    "Have you seen the bread?",
    "How much am I paying? I'm paying... one point five million and change.",
    "Fuck you man!  You don't like my fucking music get your own fucking cab!",
    "Hello! How are you?",
]


def get_pkl(filepath):
    """Load pkl files."""
    filename = os.path.join(DIRNAME, filepath)
    return joblib.load(filename)


def tranform_our_string(our_string):
    """Delete trash from string."""
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
    """Our int main."""
    vectorizer = get_pkl('news_vectorizer_dump_0_1.pkl')
    our_model = get_pkl('news_model_dump_0_1.pkl')
    our_text = EXAMPLE_TEXT[0]
    st.set_page_config(
        page_title="EKh Made2020",
    )
    st.markdown("<h1 style='text-align: center;'>Application for finding aggressive people</h1>",
                unsafe_allow_html=True)
    lenght_example_text = len(EXAMPLE_TEXT)
    random_index = random.randint(0, lenght_example_text - 1)

    if st.button('Random example'):
        our_text = EXAMPLE_TEXT[random_index]
        st.text_input(
            TEXT_FROM_CURRENT_PERSON,
            value='',
        )
    else:
        our_text = st.text_input(TEXT_FROM_CURRENT_PERSON)

    our_text_after_tranform = tranform_our_string(our_text)

    class_index = our_model.predict(
        vectorizer.transform([our_text_after_tranform]))[0]
    st.text("Current text:")
    st.markdown(f"<p><b>{our_text}</b></p>",
                unsafe_allow_html=True)
    st.text("Current situation:")

    if class_index:
        st.markdown("<h1 style='text-align: center; color: red;'>ALARM!!!</h1>",
                    unsafe_allow_html=True)

    else:
        st.markdown("<h1 style='text-align: center; color: green;'>Normal situation</h1>",
                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()
