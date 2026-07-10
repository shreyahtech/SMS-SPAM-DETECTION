import streamlit as st
import pickle
import string
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit UI
# Streamlit UI
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Classifier")

st.markdown(
    """
    <div style="
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #d9d9d9;
    ">
        <h4>Machine Learning based SMS Classification System</h4>
        <p>
        An NLP-powered application that uses TF-IDF vectorization
        and Multinomial Naive Bayes to classify SMS messages as
        spam or legitimate.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

input_sms = st.text_area("Enter your message")


if st.button("Predict"):

    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("Spam Message")
    else:
        st.success("Not Spam")