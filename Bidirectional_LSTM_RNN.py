import pandas as pd

df = pd.read_csv('/content/drive/My Drive/fake-news/train.csv')

df = df.dropna()

X = df.drop('label', axis = 1)
y = df['label']


import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional

voc_size = 10000

messages = X.copy()

messages.reset_index(inplace = True)

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()

corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
onehot_repr = [one_hot(words, voc_size) for words in corpus] 

sent_length = 20

embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = sent_length)

embedding_vector_features = 40 

model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length = sent_length))  
model.add(Dropout(0.3)) 
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3)) 
model.add(Dense(1, activation = 'sigmoid')) 
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

import numpy as np

X_final = np.array(embedded_docs)
y_final = np.array(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.33, random_state = 42)

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 20, batch_size = 64)

y_pred = model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
 
cm = confusion_matrix(y_test, y_pred) 
score = accuracy_score(y_test, y_pred)
 
print(cm) 
print(score)  
