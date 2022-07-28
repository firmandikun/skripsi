# IMPORT LIBRARY
import pandas as pd
import numpy as np
import streamlit as st
import pickle as pkl

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
text = text.lower()
submit = st.button("Submit")

## SAVE INPUT IN DATAFRAME
data_result = pd.DataFrame({'Text':[text]})

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

