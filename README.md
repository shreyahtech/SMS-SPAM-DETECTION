# SMS Spam Detection

A machine learning web application that classifies SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and Streamlit.

## Features

- SMS text classification
- TF-IDF Vectorization
- Multinomial Naive Bayes classifier
- Interactive Streamlit web interface
- Real-time prediction

## 🛠 Tech Stack

- Python
- Scikit-learn
- Pandas
- NLTK
- Streamlit
- Pickle

## Project Structure

```
SMS-SPAM-DETECTION/
│── app.py
│── model.pkl
│── vectorizer.pkl
│── sms_spam_detection.ipynb
│── requirements.txt
│── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Demo

Enter an SMS message and the app predicts whether it is **Spam** or **Not Spam**.

## Author

**Shreya Hasija**

https://shreyahtech-sms-spam-detection.streamlit.app/
