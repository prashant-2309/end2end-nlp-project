# Core Pkgs
import streamlit as st
import altair as alt
import plotly.express as px
import pickle
import string
from PIL import Image
from nltk.corpus import stopwords
import nltk
import requests
from nltk.stem.porter import PorterStemmer
from pathlib import Path
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

st.set_page_config(page_title="Text Classifiers")

def main():
    st.title("Text Classifier App")
    menu = ["Movies Recommendation", "Emotion Dection", "Email/SMS Spam Detection","About"]
    choice = st.sidebar.selectbox("Menu",menu)
#-----------------------------------------------------------------------------------------------------------------------------------
    if choice =="Emotion Dection":
        def predict_emotions(docx):
            p1 = Path(__file__).parent / "/app/end2end-nlp-project/App/models/emotion.pkl"
            f1 = open(p1, "rb")
            pipe_lr = joblib.load(f1)
            results = pipe_lr.predict([docx])
            return results[0]

        def get_prediction_proba(docx):
            pipe_lr = joblib.load(f1)
            results = pipe_lr.predict_proba([docx])
            return results

        emotions_emoji_dict = {"anger": "😠", "fear": "😨😱", "joy": "😃", "love": "❤️", "sadness": "😔", "surprise": "😮"}

        pipe_lr = joblib.load(f1)
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
# -----------------------------------------------------------------------------------------------------------------------------------
    elif choice =="Email/SMS Spam Detection":
        st.subheader("Email/SMS Spam ")
        tfidf = pickle.load(open('/app/end2end-nlp-project/App/models/vectorizer.pkl', 'rb'))
        model = pickle.load(open('/app/end2end-nlp-project/App/models/model.pkl', 'rb'))
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
# -----------------------------------------------------------------------------------------------------------------------------------

    elif choice =="Movies Recommendation":
        st.subheader("Movies Recommendation")

        def fetch_poster(movie_id):
            url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(
                movie_id)
            data = requests.get(url)
            data = data.json()
            poster_path = data['poster_path']
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path

        def recommend(movie):
            index = movies[movies['title'] == movie].index[0]
            distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
            recommended_movie_names = []
            recommended_movie_posters = []
            for i in distances[1:6]:
                # fetch the movie poster
                movie_id = movies.iloc[i[0]].movie_id
                recommended_movie_posters.append(fetch_poster(movie_id))
                recommended_movie_names.append(movies.iloc[i[0]].title)

            return recommended_movie_names, recommended_movie_posters


        movies = pd.read_pickle(open("/app/end2end-nlp-project/App/models/movie_list.pkl", "rb"))
        similarity = pickle.load(open("/app/end2end-nlp-project/App/models/similarity.pkl", "rb"))

        movie_list = movies['title'].values
        selected_movie = st.selectbox(
            "Type or select a movie from the dropdown",
            movie_list
        )

        if st.button('Show Recommendation'):
            recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_movie_names[0])
                st.image(recommended_movie_posters[0])
            with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])

            with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
            with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
            with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4])

        st.write("Due to the available dataset contains data before year 2018, all the recommendation will be till the year 2018")




# -----------------------------------------------------------------------------------------------------------------------------------
    else:
        
        st.subheader("Made by Prashant Singh")
        # image = Image.open('/app/end2end-nlp-project/App/damian.jpg')
        # st.image(image)

if __name__ =='__main__':
    main()
