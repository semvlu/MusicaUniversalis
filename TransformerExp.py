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
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model


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
