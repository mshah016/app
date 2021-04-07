#dependencies
import tensorflow as tf
import keras
from keras.layers import Input, LSTM, Embedding, Bidirectional, GRU
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import sqlalchemy as s 

def importData():
    """import data from your training source and separate into appropriate lists"""
    # import data
    review_data = pd.read_csv("training_data/labeled_responses.csv")
    # print(review_data)

    # separate into training and testing data 
    train_data, test_data = review_data[:13000], review_data[13000:]

    # initialize lists to hold data 
    training_sentences = []
    training_labels = []

    test_sentences = []
    test_labels = []

    # declare with row has sentiment and which row has labels. append to appropriate list 
    for index, row in train_data.iterrows():
        sent = row[9]
        lab = int(row[11])
        training_sentences.append(str(sent))
        training_labels.append(lab)
        
    for index, row in test_data.iterrows():
        sent = row[9]
        lab = int(row[11])
        test_sentences.append(str(sent))
        test_labels.append(lab)
        
    # convert labels into numpy array for feeding into model 
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(test_labels)

    # training_sentences[0]
    # training_labels[0]
    return training_sentences, test_sentences, test_labels, training_labels_final







def Tokenization(): 
    """take in data lists from importData() function and tokenize (vectorize) them, then convert each list to the same length"""
    vocab_size = 10000
    embedding_dim = 16
    max_length = 100
    trunc_type = 'post'
    oov_tok = '<OOV>'

    training_sentences, test_sentences, test_labels, training_labels_final = importData()
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

    # return tokenizer to use when making predictions, return padded to run in model 
    return padded, tokenizer








def Model(inputText):
    """define Sequential model and make predictions on the data"""
    # Set Parameters
    vocab_size = 10000
    embedding_dim = 16
    max_length = 100

    # Define Model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length, embeddings_regularizer = keras.regularizers.l2(.001)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate = 0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train Model
    num_epochs = 10
    batch_size = 128

    # import variables from previous functions
    padded, tokenizer = Tokenization() 
    _, _, _, training_labels_final = importData()

    # Fit model on data 
    model.fit(padded, training_labels_final, batch_size = batch_size, epochs=num_epochs, validation_split=0.2, shuffle=True)
    print(model.summary())

    # Predictions
    pred = ''
    pred_score = 0.00
    prediction_list = []
    pred_score_list = []
    inputList = [inputText]
    for i in range(0, len(inputList)):
        max_length = 100
        trunc_type = 'post'
        sequence = tokenizer.texts_to_sequences(inputList)
        padded = pad_sequences(sequence, maxlen = max_length, truncating = trunc_type)
        output = model.predict(padded)
        if (output[i][0] <= 0.5):
            # print(output[i])
            pred_score = output[i][0]
            pred = 'negative'
            prediction_list.append(pred)
            pred_score_list.append(pred_score)
        else:
            # print(output[i])
            pred_score = output[i][0]
            pred = 'positive'
            prediction_list.append(pred)
            pred_score_list.append(pred_score)

        print('Review: ' + inputList[i] + '\n' + 'Sentiment: ' + pred + ' ' + str(output[i][0]) + '\n' + '\n')
    return pred, prediction_list, pred_score_list


# # # # # Test Model # # # # #
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

#print(Model(inputText))

