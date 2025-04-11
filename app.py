## Step 1: import  the  essential libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing  import sequence
from tensorflow.keras.models  import load_model

## import  imdb dataset  and word index
word_index =   imbd.get_word_index()
revers_word_index = {value: key for  key,value in word_index.items()}

## Load the pre-trained model  with  ReLU activation
model =  load_model('imbd_rnn_model.h5')

###  Step 2:  Helper Functions
##  function  to decode  reveiws
def  decode_reveiw(text):
    return ' '.join([revers_word_index.get(i- 3,'?')for i  in  text])
##  function to process user input
def  preprocess_text(text):
    words =  text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

import  streamlit  as  st
## streamlit app
st.write('Movie Reveiw  Sentiment Analysis')
st.input_text("Enter the movie review")
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    ## make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
elif  st.input_text  in None:
    st.warn('Please enter a movie review.')