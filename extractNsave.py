import numpy as np
import pandas as pd
import pretty_midi
import pickle
# extract_feat for CNN & LSTM. SVM is raveled
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

    with open ('rolls.pickle', 'wb') as f:
        pickle.dump(features, f)

    return features

'''
with open('features.pickle', 'rb') as f:
    feat = pickle.load(f)

print(feat.head())  # print the first few rows of the DataFrame
print(feat.info())  # print information about the DataFrame
print(feat.describe())
'''