import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

st.title("Sentiment Prediction App")
DATA_PATH = "sentiment_data.csv" 

try:
    data = pd.read_csv(DATA_PATH)
    data.dropna(subset=['Comment'], inplace=True)
    x = data['Comment']
    y = data['Sentiment']
    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    x_train_tfidf = vectorizer.fit_transform(xtrain)
    x_test_tfidf = vectorizer.transform(xtest)

    clf = RandomForestClassifier(class_weight='balanced', max_depth=10, n_estimators=100, random_state=42)
    clf.fit(x_train_tfidf, ytrain)

    user_input = st.text_area(" Enter a comment to predict sentiment:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning(" Please enter some text.")
        else:
            input_vector = vectorizer.transform([user_input])
            prediction_num = clf.predict(input_vector)[0]
            prediction = label_map.get(prediction_num, prediction_num)
            emoji_map = {
                "Positive": "😊",
                "Negative": "😞",
                "Neutral": "😐"
            }
            emoji = emoji_map.get(prediction, "❓")

            st.success(f"Predicted Sentiment: **{prediction}** {emoji}")

except FileNotFoundError:
    st.error("❌ The file 'sentiment_data.csv' was not found. Please make sure it exists in the same folder as this script.")
