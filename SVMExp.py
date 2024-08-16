import numpy as np
import pandas as pd
from sklearn.svm import SVC
import sklearn
import pretty_midi
import os
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def extract_feat(metadata):
    features = []
    for index, row in metadata.iterrows(): # extract data from metadata
        file = pretty_midi.PrettyMIDI(row.midi_filename) # read midi file 

        roll = file.get_piano_roll(fs=5)
        roll = roll[21:109, :150] # sampling 0-30s, 30*5=150
        roll = np.ravel(roll)
        roll = roll.astype(int)

        chroma = file.get_chroma(fs=5)
        chroma = chroma[:, :150] # sampling 0-30s, 30*5=150
        chroma = np.ravel(chroma)
        chroma = chroma.astype(int)

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
    ''' 
    # Orig: 60 composer labels
    seen = set()
    for row in features.composer: 
        if row in seen:
            continue
        else:
            seen.add(row)
            #print(row)
    #print('Total #composer: ', len(seen))
    # 7 composers, 971 pieces   
    '''
    
    '''
    c = features.composer.value_counts()
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.plot(c,'o-')   
    for i, v in enumerate(c):
        ax.annotate(str(c[i]), xy=(i,v), xytext=(-7,7), textcoords='offset points')
    plt.xticks(rotation=45, ha='right', fontsize=8)

    ax.set_title('Works Distrinution by Composer')
    plt.show()
    '''
    return features


os.chdir('maestro-v3.0.0/')
df = pd.read_csv('maestro-v3.0.0.csv')
feat = extract_feat(df)

X_train_roll = np.array(feat['roll'].tolist())
X_train_chroma = np.array(feat['chroma'].tolist())
X_train_pre = np.concatenate((X_train_roll, X_train_chroma), axis=1)

le = LabelEncoder()
le.fit(feat['composer'])
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, le.transform(feat['composer']), 
stratify=feat['composer'], test_size=0.2, random_state=42) 

 
clf = SVC()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)


print('Accuracy: ', accuracy_score(y_test, pred))
print()
print('Precision (macro): ', precision_score(y_test, pred, average='macro'))
print('Precision (micro): ', precision_score(y_test, pred, average='micro'))
print('Precision (weighted): ', precision_score(y_test, pred, average='weighted'))
print()
print('Recall (macro): ', recall_score(y_test, pred, average='macro'))
print('Recall (micro): ', recall_score(y_test, pred, average='micro'))
print('Recall (weighted): ', recall_score(y_test, pred, average='weighted'))
print()
print('F1 Score (macro): ', f1_score(y_test, pred, average='macro'))
print('F1 Score (micro): ', f1_score(y_test, pred, average='micro'))
print('F1 Score (weighted): ', f1_score(y_test, pred, average='weighted'))

cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.xticks(rotation=45, ha='right')
plt.show()

