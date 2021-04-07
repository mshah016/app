import tensorflow as tf
import numpy as np
from statistics import mean
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf_model = tf.keras.models.load_model('models/tf_model')

tf_model.summary()


# def inputPrediction(inputText):
#     inputList = [inputText]
#     max_length = 100
#     trunc_type = 'post'
#     sequence = tokenizer.texts_to_sequences(inputList)
#     padded = pad_sequences(sequence, maxlen = max_length, truncating = trunc_type)
#     output = tf_model.predict(padded)
#     # print(output)
#     # print(max_output)

#     pred = ''
#     for i in range(0, len(inputList)):
#         if (output[i] <= 0.5):
#             pred = 'negative'
#         else:
#             pred = 'positive'

#         print('Review: ' + inputList[i] + '\n' + 'Sentiment: ' + pred + ' ' + str(output[i]) + '\n' + '\n')
#     return pred 
