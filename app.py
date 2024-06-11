import streamlit as st


st.markdown("Created on Thu June 11")
st.markdown("@author: Prit.Dhanani")




import tensorflow as tf
from tensorflow import keras
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import nltk
import re
import pickle
from transformers import pipeline





def lemetization(mess):
    corpus = []
    stem = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))
    pattern = re.compile('[^a-zA-Z]')
    pattern1 = re.compile(r'\b{}\b'.format(re.escape('br')))
                      
    for i in (range(0,len(mess))):
    #     review = re.sub('[^a-bA-B]',' ',mess[i])
        review = pattern.sub(' ', mess[i])
        cleaned_text = pattern1.sub('',review)
        review = review.lower().split()

        review = [stem.lemmatize(word) for word in review if not word in stop_words]
        review = ' '.join(review)
        corpus.append(review)
    
    return corpus



def tokenizer(text):

    toke = pickle.load(open('tokenizer.pkl','rb'))

    temp_lam = lemetization([text])

    print(temp_lam)

    temp_toke = toke.texts_to_sequences(temp_lam)

    temp_toke_pad = keras.preprocessing.sequence.pad_sequences(temp_toke, maxlen=300)

    return temp_toke_pad

def sentiment_out(text):
    
    preprocess = tokenizer(text)

    lstm = keras.models.load_model('lstm_model.h5')
    cnn = keras.models.load_model('cnn_model.h5')
    pipe = pickle.load(open('pipe.pkl','rb'))
    # pipe = pipeline('sentiment-analysis')



    pred_lstm = lstm.predict(preprocess)
    pred_cnn = cnn.predict(preprocess)
    pred_pre = pipe(text)

    pred_lstm_ = np.argmax(pred_lstm[0])
    pred_cnn_ = np.argmax(pred_cnn[0])
    pred_pre_ = list(pred_pre[0].values())[0]



    if (pred_pre_)=='POSITIVE':
        pred_pre_ = 1
    else:
        pred_pre_=0

    x = [pred_lstm_,pred_cnn_,pred_pre_]
    majority = find_majority_element(x)
    return  majority

from collections import Counter

def find_majority_element(nums):
    print(nums)
    count = Counter(nums)
    majority_element = max(count, key=count.get)
    print("Majority element of nums1:", majority_element)  
    if majority_element == 1:
        temp ='POSITIVE'
    else:
        temp="NEGATIVE"
    return temp




# print('result is :',sentiment_out('hello good to see you my friends'))


def main():

    st.title('Sentiment Analysis')
    html = ''' 

    <div stype=" background-color:tomato;padding=10px">
    <h2 style="color:white;text-align:center; ">Streamlit Sentiment Analyzer AI App </h2>       
    </div>
     '''

    st.markdown(html,unsafe_allow_html=True)

    text = st.text_area('Enter Your Review Here',height=250)
    result = ''

    if st.button('Analyze'):
        result = sentiment_out(text)
        # st.success(result)    
        # if st.button("Analyze"):
        
    
        if result == "POSITIVE":
            st.markdown(f'<div style="background-color: green; color: white; padding: 10px;">{result.upper()}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: red; color: white; padding: 10px;">{result.upper()}</div>', unsafe_allow_html=True)

    if st.button("About"):
        st.text("This work is done by Prit Dhanani")
        st.text("I try to clssify the sentiment whether it's positive or negative")
        st.text('Therefor I use LSTM model, CNN (for ebstract main feature from text) and a pre-trained Bert model')
        st.text('Then it voated out from these three model and give out result in POSITIVE or NEGATIVE')

if __name__ == '__main__':
    main()