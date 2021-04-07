#dependencies
import tensorflow as tf
import keras
from keras.layers import Input, LSTM, Embedding, Bidirectional, GRU
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import pandas as pd

review_data = pd.read_csv("data/Random Responses 3 - sorted.csv")
# print(review_data)

train_data, test_data = review_data[:13000], review_data[13000:]

training_sentences = []
training_labels = []

test_sentences = []
test_labels = []

for index, row in train_data.iterrows():
    sent = row[9]
    lab = int(row[11])
#     print(type(lab))
    training_sentences.append(str(sent))
    training_labels.append(lab)
    
for index, row in test_data.iterrows():
    sent = row[9]
    lab = int(row[11])
    test_sentences.append(str(sent))
    
test_labels.append(lab)
    
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(test_labels)

training_sentences[0]
print(type(training_sentences[0]))
training_labels[0]
print(type(training_labels[0]))



# Tokenization

# parameters
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
oov_tok = '<OOV>' #Out Of Vocabulary 

# from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# sequencing
sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences, maxlen=max_length, truncating = trunc_type)
testing_sequences = tokenizer.texts_to_sequences(test_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

# print(padded.shape)

# Model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length, embeddings_regularizer = keras.regularizers.l2(.001)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    # tf.keras.layers.Dropout(rate = 0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(rate = 0.5),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])



# Train Model

num_epochs = 10
batch_size = 128
# validation_data = (testing_padded, testing_labels_final)

history = model.fit(padded, training_labels_final, batch_size = batch_size, epochs=num_epochs, validation_split=0.2, shuffle=True)
model.summary()
# # Prediction

# new_sentences = [
#     'terrible movie',
#     'never watching this again',
#     'loved this so much, amazing!',
#     'awful movie, who produced this?',
#     'amazing addition to the franchise',
#     'pretty good, not the best but not bad either'
# ]

# new_sequences = tokenizer.texts_to_sequences(new_sentences)
# padded = pad_sequences(new_sequences, maxlen = max_length, truncating = trunc_type)
# output = model.predict(padded)


# # prediction = 'negative' if output <= 0.5 else 'positive'
# prediction = ''
# for i in range(0, len(new_sentences)):
#     if output[i] <= 0.5:
#         prediction = 'negative'
#     else:
#         prediction = 'positive'
#     print('Review: ' + new_sentences[i] + '\n' + 'Sentiment: ' + prediction + ' ' + str(output[i]) + '\n' + '\n')


def inputPrediction(inputText):
    pred = ''
    inputList = inputText
    for i in range(0, len(inputList)):
        max_length = 100
        trunc_type = 'post'
        sequence = tokenizer.texts_to_sequences(inputList)
        padded = pad_sequences(sequence, maxlen = max_length, truncating = trunc_type)
        output = model.predict(padded)
        if (output[i][0] <= 0.5):
            print(output[i])
            pred = 'negative'
        else:
            print(output[i])
            pred = 'positive'

        print('Review: ' + inputList[i] + '\n' + 'Sentiment: ' + pred + ' ' + str(output[i][0]) + '\n' + '\n')
    return pred

inputText = [
    "I need to cancel my order.",
    "how do I refer a friend?",
    "Good morning. I have an order that I placed that I need to change the shipping address. Can you help?",
    "Interested in the white sole fish pink shrimp",
    "How can I find out the status of an order?",
    "how do I get to the wholesale site",
    "thanks so much! this was such a great experience, really enjoyed it",
    "keep up the good work, loved it!",
    "this was awful, never doing this again",
    "thanks for nothing, absolutely not, do not recommend"
]

# print(inputPrediction(inputText))
# print(inputText[3])
