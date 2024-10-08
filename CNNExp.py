from extract import extract_feat
import numpy as np
import pandas as pd
import pretty_midi
import os
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from scores import scores_cm

os.chdir('maestro-v3.0.0/')
df = pd.read_csv('maestro-v3.0.0.csv')
feat = extract_feat(df)


print(feat['roll'].iloc[1].shape)
print(feat['roll'].iloc[1][1].shape)
print(feat['chroma'].iloc[1].shape)
#X_train_roll = np.array(feat['roll'].tolist())
#X_train_chroma = np.array(feat['chroma'].tolist())
#X_train_pre = np.concatenate((X_train_roll, X_train_chroma),axis=1) # axis=0
X_train_pre = []
for i in range(len(feat['roll'])):
    X_train_pre.append(np.concatenate((feat['roll'].iloc[i], feat['chroma'].iloc[i]), axis=0))
X_train_pre = np.array(X_train_pre)


le = LabelEncoder()
le.fit(feat['composer'])
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, le.transform(feat['composer']), 
stratify=feat['composer'], test_size=0.2, random_state=42) 




model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100,150,1), dtype='float32'),

    tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling2D(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), # multi classes, int lbl
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)
pred = model.predict(X_test)

print(pred)
scores_cm(pred=pred, y_test=y_test, le=le)
