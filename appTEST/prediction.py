#load model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import model as m

#load model
model = tf. keras.models.load_model('saved_model/TGA_model')

# Check its architecture
model.summary()


# make predictions
def inputPrediction(inputText):

    # import tokenizer from previous model.py
    _, tokenizer = m.Tokenization()
    pred = ''
    confidence = 0.0
    inputList = [inputText]
    for i in range(0, len(inputList)):
        max_length = 100
        trunc_type = 'post'
        sequence = tokenizer.texts_to_sequences(inputList)
        padded = pad_sequences(sequence, maxlen = max_length, truncating = trunc_type)
        output = model.predict(padded)
        if (output[i][0] <= 0.5):
            print(output[i])
            pred = 'negative'
            confidence += output[i][0] 
        else:
            print(output[i])
            pred = 'positive'
            confidence += output[i][0]

        print('Review: ' + inputList[i] + '\n' + 'Sentiment: ' + pred + ' ' + str(output[i][0]) + '\n' + '\n')
    return pred, confidence 

sample =  "happy good wonderful"

inputPrediction(sample)