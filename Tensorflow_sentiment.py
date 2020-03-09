from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import json
import numpy as np

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

with open("review.json", 'r') as f:
    datastore = json.load(f)
    
review = np.array([])
labels = np.array([])

for item in datastore:
    review = np.append(review, [item['comments']])
    if int(item['star_rating'])>=3:
        labels = np.append(labels,1)
    else:
        labels = np.append(labels,0)
        
review_train = review[:4000]
review_test = review[4000:]

label_train = labels[:4000]
label_test = labels[4000:]
    
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(review_train)
word_index = tokenizer.word_index

review_sequences_train = tokenizer.texts_to_sequences(review_train)
padded_train = pad_sequences(review_sequences_train, maxlen = max_length, truncating=trunc_type)

review_sequences_test = tokenizer.texts_to_sequences(review_test)
padded_test = pad_sequences(review_sequences_test, maxlen = max_length)


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])


model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
print(model.summary())

num_epochs = 10
history = model.fit(padded_train,label_train, epochs=num_epochs, validation_data=(padded_test, label_test), verbose=2)


