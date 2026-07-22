import streamlit as st
import pickle
import string
import nltk
import plotly.graph_objects as go

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- CSS ----------------

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

local_css("assets/style.css")

# ---------------- NLTK ----------------

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------- LOAD MODEL ----------------

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

ps = PorterStemmer()

# ---------------- PREPROCESSING ----------------

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
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ---------------- SPAM METER ----------------

def spam_meter(score):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Spam Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ff4da6"},
            "steps": [
                {"range": [0, 30], "color": "#241329"},
                {"range": [30, 70], "color": "#5e2a58"},
                {"range": [70, 100], "color": "#ff4da6"}
            ]
        }
    ))

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )

    return fig

# ---------------- HERO ----------------

st.markdown("""
<h1 class='hero-title'>
🛡️ SpamShield AI
</h1>

<p class='hero-subtitle'>
AI-Powered Message Intelligence • NLP • Threat Detection
</p>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------

input_sms = st.text_area(
    "✉️ Enter your message",
    height=180,
    placeholder="Type or paste an SMS message here..."
)

# ---------------- PREDICT ----------------

if st.button("Analyze Message"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
        st.stop()

    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Model probability
    probabilities = model.predict_proba(vector_input)[0]
    spam_score = float(probabilities[1] * 100)

    # ---------------- RULE-BASED BOOST ----------------

    suspicious_words = [
        "bank",
        "account",
        "verify",
        "verification",
        "login",
        "password",
        "otp",
        "click",
        "link",
        "urgent",
        "suspended",
        "reward",
        "claim",
        "winner",
        "won",
        "free",
        "cash",
        "offer",
        "limited",
        "prize",
        "congratulations",
        "gift",
        "bonus",
        "selected"
    ]

    message_lower = input_sms.lower()

    matches = sum(
        1 for word in suspicious_words
        if word in message_lower
    )

    spam_score += matches * 5
    spam_score = min(spam_score, 100)

    # Final Classification

    if spam_score >= 50:
        prediction = 1
    else:
        prediction = 0

    # ---------------- SPAM METER ----------------

    st.plotly_chart(
        spam_meter(spam_score),
        use_container_width=True
    )

    # ---------------- RESULT ----------------

    if prediction == 1:

        st.error(
            f"🚨 Spam Message Detected | Risk Score: {spam_score:.1f}%"
        )

    else:

        st.success(
            f"✅ Legitimate Message | Confidence: {(100-spam_score):.1f}%"
        )

    st.markdown("---")

    # ---------------- MESSAGE STATS ----------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Words",
        len(input_sms.split())
    )

    col2.metric(
        "Characters",
        len(input_sms)
    )

    col3.metric(
        "Spam Risk",
        f"{spam_score:.1f}%"
    )