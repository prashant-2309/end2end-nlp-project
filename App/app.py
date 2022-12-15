# Core Pkgs
import streamlit as st
import altair as alt
import plotly.express as px
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

# EDA Pkgs
import pandas as pd
import numpy as np
from datetime import datetime

# Utils
import joblib




# Track Utils
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table



#Functions below
def predict_emotions(docx):
    pipe_lr = joblib.load(open("/app/end2end-nlp-project/App/models/emotion.pkl","rb"))
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    pipe_lr = joblib.load(open("/app/end2end-nlp-project/App/models/emotion.pkl","rb"))
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":"😠", "fear":"😨😱", "joy":"🤧","love":"❤️", "sadness":"😔", "surprise":"😮"}

ps = PorterStemmer()
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

def main():
    st.title("Text Classifier App")
    menu = ["Emotion Dection", "Email/SMS Spam Detection", "About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice =="Emotion Dection":
        pipe_lr = joblib.load(open("/app/end2end-nlp-project/App/models/emotion.pkl","rb"))
        st.subheader("Emotion In Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2)

            #Applying the above functions here

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif choice =="Email/SMS Spam Detection":
        st.subheader("Email/SMS Spam ")
        tfidf = pickle.load(open('/app/end2end-nlp-project/App/models/vectorizer.pkl', 'rb'))
        model = pickle.load(open('/app/end2end-nlp-project/App/models/model.pkl', 'rb'))



        

        input_sms = st.text_area("Enter the message")

        if st.button('Predict'):

            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            # 4. Display
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")





    else:
        st.subheader("About")
        st.subheader("Made by Prashant Singh")

if __name__ =='__main__':
    main()
