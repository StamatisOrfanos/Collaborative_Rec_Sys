import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Concatenate, Input, Embedding, Flatten
from sklearn.model_selection import train_test_split


# Import the data then, get train, test sets and creating the embedding
# layer starting with the vocabulary of the layer
books = pd.read_csv("archive//books.csv")
ratings = pd.read_csv("archive//ratings.csv")

book_ids = ratings['book_id'].nunique()
user_ids = ratings['user_id'].nunique()


# The book embedding (1D array of book_ids)
# We want to pass the output of the Input layer to the Embedding layer
input_books = Input(shape=[1])
embed_books = Embedding(book_ids + 1, 256)(input_books)
books_out = Flatten()(embed_books)


# The user embedding
input_users = Input(shape=[1])
embed_users = Embedding(user_ids + 1, 256)(input_users)
users_out = Flatten()(embed_users)


# Define the model
input_layer  = Concatenate(axis=1)([books_out, users_out])
hidden_layer1 = Dense(units=512, activation='relu')(input_layer)
hidden_layer2 = Dense(units=256, activation='softmax')(hidden_layer1)
output_layer = Dense(units=5, activation='softmax')(hidden_layer2)

model = tfk.Model(inputs=[input_books, input_users], outputs=output_layer)
print(model.summary())

model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])
hist = model.fit([ratings.book_id, ratings.user_id], ratings.rating, batch_size=256, epochs=6,
                    verbose=1, validation_split = 0.1)


# Lastly save the model to be used in the js application in a Tensorflow-Keras mode
model.save('model.h5')
print("Model successfully saved")


# Use the data to make JSON files to use for the application
web_book_data = books[["book_id", "title", "image_url", "authors"]]
web_book_data = web_book_data.sort_values('book_id')
web_book_data.to_json(r'web_book_data.json', orient='records')