# IMPORT LIBRARY
import pandas as pd
import numpy as np
import streamlit as st
import pickle as pkl
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# LOAD NAIVE BAYES MODEL & TF-IDF VECTORIZATION
filename_model = 'finalized_model_naivebayes.sav'
filename_tfidf = 'vectorizer.pickle'
# model = pkl.load(open(filename, 'rb'))
model = pkl.load(open(filename_model, 'rb'))
vect = pkl.load(open(filename_tfidf, 'rb'))

# CREATE WEB APP
## SET PAGE
st.set_page_config(page_title="Sentiment Analysis Web App", page_icon=":page_facing_up:", layout="centered")

## SET TITLE
st.write('')
st.markdown('<div style="text-align: justify; font-size:300%"> <b>Web App Analisis Sentimen</b> </div>',
            unsafe_allow_html=True)

## SET PICTURE
left_column_chart_row4, mid_column_chart_row4, right_column_chart_row4 = st.columns([2,4,2])
mid_column_chart_row4.image("aset_sentimen.png", use_column_width=True)

## SET DESCRIPTION
st.markdown('<div style="text-align: justify; font-size:160%"> Web App ini merupakan suatu aplikasi di mana kita bisa memprediksi/mengklasifikasi sentimen suatu teks. Model klasifikasi yang digunakan pada analisis sentimen ini yaitu Naive Bayes.</div>',
            unsafe_allow_html=True)
st.write('#### Developer : M. Firman Setiawan')

## ADD HORIZONTAL LINE
st.markdown("""---""")

## ADD TEXT INPUT & SUBMIT BUTTON
text = st.text_input('Masukkan kalimat yang akan dianalisis sentimennya', placeholder='Contoh : COVID-19 yang melanda negeri ini menguntungkan bagi beberapa pihak')
submit = st.button("Submit")

## SAVE INPUT IN DATAFRAME
data_result = pd.DataFrame({'Text':[text]})

## TEXT PREPROCESSING
data_result['Text'] = data_result['Text'].str.lower()

def remove(text):
  text = text.replace('\\t', ' ').replace('\\n', ' ').replace('\\u', ' ').replace('\\', ' ')
  text = text.encode('ascii', 'replace').decode('ascii')
  text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", ' ', text).split())
  return text.replace('http://', ' ').replace('https://', ' ')
data_result['Text'] = data_result['Text'].apply(remove)

def remove_number(text):
  return re.sub(r'\d+', '', text)
data_result['Text'] = data_result['Text'].apply(remove_number)

def remove_punc(text):
  return text.translate(str.maketrans('','',string.punctuation))
data_result['Text'] = data_result['Text'].apply(remove_punc)

def remove_whitespace(text):
  return text.strip()
data_result['Text'] = data_result['Text'].apply(remove_whitespace)

def remove_whitespace_multi(text):
  return re.sub('\s+', ' ', text)
data_result['Text'] = data_result['Text'].apply(remove_whitespace_multi)

def remove_single_char(text):
  return re.sub(r'\b[a-zA-Z]\b', '', text)
data_result['Text'] = data_result['Text'].apply(remove_single_char)

nltk.download('punkt')
def word_tokenize_wrapper(text):
  return word_tokenize(text)
data_result['Text'] = data_result['Text'].apply(word_tokenize_wrapper)

nltk.download('stopwords')
list_stopwords = stopwords.words('indonesian')
list_stopwords = set(list_stopwords)
def remove_stopwords(words):
  return [word for word in words if word not in list_stopwords]
data_result['Text'] = data_result['Text'].apply(remove_stopwords)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemming(words):
  return [stemmer.stem(word) for word in words]
data_result['Text'] = data_result['Text'].apply(stemming)

data_result['Text'] = data_result['Text'].agg(lambda x: ','.join(map(str, x)))

## SENTIMENT ANALYSIS
y_pred = model.predict(vect.transform(data_result['Text'].values))
y_pred_proba = model.predict_proba(vect.transform(data_result['Text'].values))

## DISPLAY OUTPUT
if submit:
    if y_pred == 1:
        result = 'Kalimat di atas memiliki sentimen POSITIF dengan probabilitas ' + str(np.round(np.max(y_pred_proba, axis=1), 2))[1:5] + '%'
        st.success(result)
    elif y_pred == 2:
        result = 'Kalimat di atas memiliki sentimen NEGATIF dengan probabilitas ' + str(np.round(np.max(y_pred_proba, axis=1), 2))[1:5] + '%'
        st.error(result)
    else:
        result = 'Kalimat di atas memiliki sentimen NETRAL dengan probabilitas ' + str(np.round(np.max(y_pred_proba, axis=1), 2))[1:5] + '%'
        st.warning(result)
