# Text Classification from Scratch
Project Overview
This project demonstrates text sentiment classification starting from raw text files, specifically using the IMDB movie review dataset. The implementation leverages the TextVectorization layer for word splitting and indexing, and uses a simple 1D convolutional neural network (CNN) for classification. The project is optimized for GPU acceleration.

Authors: Mark Omernick, Francois Chollet
Date Created: 2019/11/06
Last Modified: 2020/05/17
Table of Contents
Introduction
Setup
Data Preparation
Data Loading
Data Preprocessing
Model Building
Training
Evaluation
End-to-End Model
Introduction
This example shows how to perform text classification starting from raw text files on disk. We demonstrate the workflow using the unprocessed IMDB sentiment classification dataset and employ the TextVectorization layer for text preprocessing.

Setup
To ensure the proper backend is used, set the Keras backend to TensorFlow:

python
Copy code
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers
Data Preparation
First, download and extract the IMDB dataset:

shell
Copy code
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
Data Loading
Inspect the dataset structure:

shell
Copy code
ls aclImdb
ls aclImdb/test
ls aclImdb/train
Remove the unsupervised training data:

shell
Copy code
rm -r aclImdb/train/unsup
Data Preprocessing
Generate labeled datasets using text_dataset_from_directory:


batch_size = 32
raw_train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
)
raw_val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)
raw_test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)
Data Preprocessing
Define a custom standardization function to clean the text:

python
Copy code
import string
import re

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )
Create a TextVectorization layer and adapt it to the training data:

python
Copy code
max_features = 20000
embedding_dim = 128
sequence_length = 500

vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)
Vectorize the data:

python
Copy code
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)
Model Building
Build a simple 1D convolutional neural network:

python
Copy code
inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = keras.Model(inputs, predictions)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
Training
Train the model:

python
Copy code
epochs = 3
model.fit(train_ds, validation_data=val_ds, epochs=epochs)
Evaluation
Evaluate the model on the test set:

python
Copy code
model.evaluate(test_ds)
End-to-End Model
Create an end-to-end model for processing raw strings:

python
Copy code
inputs = keras.Input(shape=(1,), dtype="string")
indices = vectorize_layer(inputs)
outputs = model(indices)

end_to_end_model = keras.Model(inputs, outputs)
end_to_end_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

end_to_end_model.evaluate(raw_test_ds)
