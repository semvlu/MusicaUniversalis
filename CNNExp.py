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

def extract_feat(metadata):
    features = []
    for index, row in metadata.iterrows(): # extract data from metadata
        file = pretty_midi.PrettyMIDI(row.midi_filename) # read midi file 

        roll = file.get_piano_roll(fs=5)
        roll = roll[21:109, :150] # sampling 0-30s, 30*5=150
        #roll = np.ravel(roll)
        roll = roll.astype(int)
#should not ravel the roll
        chroma = file.get_chroma(fs=5)
        chroma = chroma[:, :150] # sampling 0-30s, 30*5=150
        #chroma = np.ravel(chroma)
        chroma = chroma.astype(int)
#should not ravel the chroma

        composer = row.canonical_composer #composer
        if('/' in composer): # work by multiple composer, e.g. 'Sergei Rachmaninoff / György Cziffra'
            collab = composer.split('/') # ['Sergei Rachmaninoff ', ' György Cziffra']
            collab[0] = collab[0][:-1]
            collab[1] = collab[1][1:]
            # [roll, chroma, collab[0]]
            features.append([roll, chroma, collab[0]])
            features.append([roll, chroma, collab[1]])
        else:
            features.append([roll, chroma, composer])
        # 54 composers

    features = pd.DataFrame(features, columns = ['roll', 'chroma', 'composer'])
    # sort by composer name   
    features = features.sort_values(by='composer')
    '''
    print('check if sorted: ')
    print(features['composer'])
    '''
    # retain composers w/ >= 50 pieces
    c = features.composer.value_counts()
    features = features[features.composer.isin(c.index[c.ge(50)])]

    return features



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
    #tf.keras.layers.Input(shape=(15000,), dtype='float32'), # 15000 = 88*150 +12*150
    tf.keras.layers.Input(shape=(100,150,1), dtype='float32'),

    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
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


ttlpred = []
for i in pred:
    maxidx = i.argmax()
    ttlpred.append(maxidx)

print('Accuracy: ', accuracy_score(y_test, ttlpred)) 
print()
print('Precision (macro): ', precision_score(y_test, ttlpred, average='macro'))
print('Precision (micro): ', precision_score(y_test, ttlpred, average='micro'))
print('Precision (weighted): ', precision_score(y_test, ttlpred, average='weighted'))
print()
print('Recall (macro): ', recall_score(y_test, ttlpred, average='macro'))
print('Recall (micro): ', recall_score(y_test, ttlpred, average='micro'))
print('Recall (weighted): ', recall_score(y_test, ttlpred, average='weighted'))
print()
print('F1 Score (macro): ', f1_score(y_test, ttlpred, average='macro'))
print('F1 Score (micro): ', f1_score(y_test, ttlpred, average='micro'))
print('F1 Score (weighted): ', f1_score(y_test, ttlpred, average='weighted'))

cm = confusion_matrix(y_test, ttlpred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.xticks(rotation=45, ha='right')
plt.show()
