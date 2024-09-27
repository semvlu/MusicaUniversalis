from extract import extract_feat
import numpy as np
import pandas as pd
import pretty_midi
import os
import warnings
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from scores import scores_cm

os.chdir('maestro-v3.0.0/')
with open('rolls.pickle', 'rb') as f:
    feat = pickle.load(f)

print(feat['roll'].iloc[1].shape)
print(feat['roll'].iloc[1][1].shape)
print(feat['chroma'].iloc[1].shape)
X_train_roll = np.array(feat['roll'].tolist())
X_train_chroma = np.array(feat['chroma'].tolist())
#X_train_pre = np.concatenate((X_train_roll, X_train_chroma),axis=1) # axis=0
X_train_pre = []
for i in range(len(feat['roll'])):
    X_train_pre.append(np.concatenate((feat['roll'].iloc[i], feat['chroma'].iloc[i]), axis=0))
X_train_pre = np.array(X_train_pre)
X_train_pre = np.transpose(X_train_pre, (0, 2, 1))

le = LabelEncoder()
le.fit(feat['composer'])
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, le.transform(feat['composer']), 
stratify=feat['composer'], test_size=0.2, random_state=42) 



# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    norm1 = LayerNormalization(epsilon=1e-4)(inputs)
    atn_out = MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads, 
        dropout=dropout)(norm1, norm1) # query & key
        # Query (Q): the context in which the attention is being computed. i.e. the input seq 
        # Key (K): the input seq that the model is attending to, to compute the attention weights.
        # Value (V): the input seq that the model is trying to weight or attend to.
    # Dropout and skip connection
    atn_out = Dropout(dropout)(atn_out)
    # Residual
    atn_res = atn_out + inputs

    norm2 = LayerNormalization(epsilon=1e-4)(atn_res)
    # Feed-forward layer
    ff = Dense(ff_dim, activation='relu')(norm2)
    ff = Dropout(dropout)(ff)
    ff = Dense(inputs.shape[-1])(ff) # 100, same as [2] 
    # Skip connection
    return ff + atn_res


# Define model
inputs = Input(shape=(150,100))

# Transformer Encoder Block
encoder_out = transformer_encoder(inputs, head_size=128, num_heads=8, ff_dim=128, dropout=0.1)
pool = GlobalAveragePooling1D()(encoder_out)
dropout_out = Dropout(0.2)(pool)
output = Dense(7, activation='softmax')(dropout_out)

model = Model(inputs, output)
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50)
pred = model.predict(X_test)
print(pred)

scores_cm(pred=pred, y_test=y_test, le=le)
