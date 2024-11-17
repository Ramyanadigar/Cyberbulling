import folium
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from lib_file import lib_path
import os
import re
import contractions
from textblob import TextBlob
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from tqdm import tqdm
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))


model = load_model("models/ConvolutionalLongShortTermMemory_model.h5",compile=False)


class_labels = ['age', 'ethnicity', 'gender', 'not_cyberbullying', 'religion']


import pickle
with open(file="models/tokens.pkl",mode="rb") as file:
    tok = pickle.load(file=file)


def clean_text(text):
    # expand contraction for words
    text=contractions.fix(text)
    # remove charectir emojes
    emoticons = [r':\)', r':\(', r':P']
    pattern = '|'.join(emoticons)
    text = re.sub(pattern, '', text)
    # remove mentions (@)
    text = re.sub(r'@\w+', '', text)
    # remove hashtags (#)
    text = re.sub(r'#\w+', '', text)
    # remove URLs (http and https)
    text = re.sub(r'https?://\S+', '', text)
    # remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    # remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Perform lemmatization on each word in the sentence
    blob = TextBlob(text)
    lemmatized_words = [word.lemmatize() for word in blob.words]
    # Join the lemmatized words back into a sentence
    lemmatized_sentence = " ".join(lemmatized_words)
    # convert to lowercase
    text = lemmatized_sentence.lower()
    return text


def cyberBullyingLocation():
    filename = "static/input_file/selected_file.csv"

    df = pd.read_csv(filename)

    # print("user_input_path : ", user_input_path)
    # df = pd.read_csv(os.path.join("static", "input_file", f"{user_input_path}"))
    # df.head(10)


    cleaned_samples = []
    for sample in tqdm(df['Text'].values):
        cleaned_samples.append(clean_text(sample))


    useful_data = []
    for cleaned_ in cleaned_samples:
        num_data=tok.texts_to_sequences([cleaned_])
        pad_text=pad_sequences(sequences=num_data,maxlen=40,padding="post",truncating="post")
        useful_data.append(pad_text)


    RESULTS = []

    for usefule_sample in useful_data:
        prediction = model.predict(usefule_sample)
        predicted_label = class_labels[np.argmax(prediction)]
        RESULTS.append(predicted_label)


    df["RESULT"] = RESULTS
    df.to_csv("static/result_file/result.csv", index=False)
    # df.head()


    df['COLORS'] = df['RESULT'].apply(lambda x: "green" if x == 'not_cyberbullying' else "red")
    # df.head()


    COMEPLETE_DETAILS = []

    for i in range(len(df)):
        cur_df = df.iloc[[i]]
        cit_name = cur_df["City"].values[0]
        result = cur_df["RESULT"].values[0]
        infos = f"CITY:{cit_name} | RESULT:{result}"
        COMEPLETE_DETAILS.append(infos)
    df['INFOS'] = COMEPLETE_DETAILS


    world_all_cities_colored = folium.Map(zoom_start=2,
                                        location=[13.133932434766733, 16.103938729508073])

    for _, city in df.iterrows():
        folium.Marker(location=[city['Lat'], city['Lng']],
                    tooltip=city['INFOS'],
                    icon=folium.Icon(color=city['COLORS'], prefix='fa', icon='circle')).add_to(world_all_cities_colored)
        
    # Render the Folium map as a string
    folium_map_html = world_all_cities_colored._repr_html_()

    return folium_map_html

