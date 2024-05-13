import pandas as pd
import os
import pretty_midi
from mido import MidiFile
import librosa

import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tslearn.metrics import dtw

import seaborn as sns
import matplotlib.pyplot as plt 


def preGram(notes):
    # get N-gram input, nt_dur = (notes, duration)
    # duration unit: fs*durCnt ms, fs=5 pro tem.
    nt_dur = []
    pos = 0
    while (pos < len(notes) - 1):
        durCnt = 1
        while (pos + 1 < len(notes) and notes[pos] == notes[pos + 1] ):
            pos += 1
            durCnt += 1
        nt_dur.append([notes[pos], durCnt])
        pos += 1
    nt_dur = np.array(nt_dur)
    nt_dur = nt_dur[np.all(np.isfinite(nt_dur), axis=1)] # remove rows w/ nan
    return nt_dur

def nGram(preGram, n):
    # get Unigram------------------------------------------
    melodyGram = []
    rhythmGram = []
    pairGram = []
    for i in range(len(preGram)-1):
        gram_note = preGram[i+1][0] - preGram[i][0]
        dur_rt = preGram[i+1][1] / preGram[i][1]
        gram_dur = round(math.log(dur_rt, 2))
        melodyGram.append(gram_note)
        rhythmGram.append(gram_dur)
    
    for i in range(len(melodyGram)):
        pairGram.append(melodyGram[i])
    for i in range(len(rhythmGram)):
        pairGram.append(rhythmGram[i])

    #-----------------------------------------------------
    if(n == 1):
        return [melodyGram, rhythmGram, pairGram]

    if (n == 2):
        # Construct Bigram
        biMelo = []
        biRhy = []
        biPair = []
        biCnt = 0
        while (biCnt + 1 < (len(preGram) - 1)):
            biMelo.append([melodyGram[biCnt], melodyGram[biCnt+1]])
            biRhy.append([rhythmGram[biCnt], rhythmGram[biCnt+1]])
            biPair.append([melodyGram[biCnt], melodyGram[biCnt+1], rhythmGram[biCnt], rhythmGram[biCnt+1]])
            biCnt += 1
        return [biMelo, biRhy, biPair]
    
    elif (n == 3):
        # Construct Trigram
        triMelo = []
        triRhy = []
        triPair = []
        triCnt = 0

        while (triCnt + 2 < (len(preGram) - 1)):
            triMelo.append([melodyGram[triCnt], melodyGram[triCnt+1], melodyGram[triCnt+2]])
            triRhy.append([rhythmGram[triCnt], rhythmGram[triCnt+1], rhythmGram[triCnt+2]])
            triPair.append([ melodyGram[triCnt], melodyGram[triCnt+1], melodyGram[triCnt+2],
            rhythmGram[triCnt], rhythmGram[triCnt+1], rhythmGram[triCnt+2]])
            triCnt += 1
        return [triMelo, triRhy, triPair]



def extract_feat(metadata):
    features = []
    for index, row in metadata.iterrows(): # extract data from metadata
        file = pretty_midi.PrettyMIDI(row.midi_filename) # read midi file 

        roll = file.get_piano_roll(fs=5)
        roll = roll[21:109]
        chroma = file.get_chroma()

        notes = np.argmax(roll, axis=0) # 1D array
        velocity = np.max(roll, axis=0) # ibid. 1D array


        notes = np.where(notes == 0, np.nan, notes) # replace 0 with nan
        # merge as 2D array, [note, velocity], shape = (time*fs, 2)
        melody = np.stack((notes, velocity), axis=-1)
        gramIn = preGram(notes)
        biGram = nGram(gramIn, 2)
        triGram = nGram(gramIn, 3)
        
        composer = row.canonical_composer #composer
        if('/' in composer): # work by multiple composer, e.g. 'Sergei Rachmaninoff / György Cziffra'
            collab = composer.split('/') # ['Sergei Rachmaninoff ', ' György Cziffra']
            collab[0] = collab[0][:-1]
            collab[1] = collab[1][1:]
            features.append([roll, chroma, biGram, triGram, collab[0], row.midi_filename])
            features.append([roll, chroma, biGram, triGram, collab[1], row.midi_filename])
        else:
            features.append([roll, chroma, biGram, triGram, composer, row.midi_filename])
        # 54 composers

    # sort by composer name
    #idx = np.argsort(features[:,8]) #[:, x] x: clmn for composer
    #features = np.take_along_axis(features, idx[:, None], axis=0)
   
    features = pd.DataFrame(features, columns = ['roll', 'chroma', 'bigram', 'trigram', 'composer', 'midi_filename'])
    features = features.sort_values(by='composer')
    #sorted(features, key=lambda l:l[4])

    print('check if sorted: ')
    print(features['composer'])

    # retain composers w/ >= 3 pieces
    c = features.composer.value_counts()
    features = features[features.composer.isin(c.index[c.ge(3)])]
    print(c)


    # Orig: 60 composer labels
    seen = set()
    for row in features.composer: 
        if row in seen:
            continue
        else:
            seen.add(row)
            print(row)
    print('Total #composer: ', len(seen))
    # 29 composers
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




# Similarity
# for each composer x
# x: other n-gram vectors for composer works, only takes 'train'
# y: N-gram vector to be evaluated, from 'test' & 'validation'

def similarity(work, featSet):
    '''
    for i in featSet:
        if ((work[8]==i[8]) and (work[9]==i[9])):
            # Create a boolean mask to select rows that are not equal to b
            mask = np.any(featSet != i[np.newaxis, :], axis=1)
            # Use the mask to select the rows that should be kept
            featSet = featSet[mask]
            break
    '''
    #featSet = [i for idx, i in featSet.iterrows() if (work['midi_filename'] != i['midi_filename']) ] # identify by midi_filename
    featSet = featSet.drop(featSet[featSet['midi_filename'] ==  work['midi_filename']].index)   

    clsComposer = 0
    clsBiSim = []
    clsTriSim = []
    clsCnt = 0
    res = []
    for idx, i in featSet.iterrows():
        curComposer = i['composer']
        
        if(clsComposer == 0):
            clsComposer = curComposer
        else: # liquidate class similarity and move clsComposer to a new one
            if(clsComposer != curComposer):
                clsBiSim = np.array(clsBiSim)
                clsTriSim = np.array(clsTriSim)
                
                # Melo: 0+3x, Rhy:1+3x, Pair:2+3x
                biMeloSim=np.array([0.0,0.0,0.0])
                cnt=0
                for j in clsBiSim[0::3]:
                    biMeloSim += j
                    cnt+=1
                biMeloSim /= cnt

                biRhySim=np.array([0.0,0.0,0.0])
                cnt=0
                for j in clsBiSim[1::3]:
                    biRhySim += j
                    cnt+=1
                biRhySim /= cnt

                biPairSim=np.array([0.0,0.0,0.0])
                cnt=0
                for j in clsBiSim[2::3]:
                    biPairSim += j
                    cnt+=1
                biPairSim /= cnt



                # triSim
                # Melo: 0+3x, Rhy:1+3x, Pair:2+3x
                triMeloSim=np.array([0.0,0.0,0.0])
                cnt=0
                for j in clsTriSim[0::3]:
                    triMeloSim += j
                    cnt+=1
                triMeloSim /= cnt

                triRhySim=np.array([0.0,0.0,0.0])
                cnt=0
                for j in clsTriSim[1::3]:
                    triRhySim += j
                    cnt+=1
                triRhySim /= cnt

                triPairSim=np.array([0.0,0.0,0.0])
                cnt=0
                for j in clsTriSim[2::3]:
                    triPairSim += j
                    cnt+=1
                triPairSim /= cnt


                #clsBiSim = clsBiSim.mean(axis=0)
                #clsBiSim = clsTriSim.mean(axis=0)
                res.append([clsComposer, [biMeloSim,biRhySim,biPairSim], [triMeloSim, triRhySim, triPairSim]])#, clsTriSim]) # class liquidation
                clsComposer = curComposer
                clsBiSim = []
                clsTriSim = []
                clsCnt = 0

        clsCnt +=1



        # Cosine similarity: cSim. A•B / (||A|| * ||B||)        
        # Euclidean distance: eucSim
        x = i['bigram'] # [n]: n is clmn for N-gram vector:2
        y = work['bigram']

        if(len(x[0]) < len(y[0])):
            # sim for bigram
            for j in range(len(x)): # 3 iteration for different feat: melody, rhythm, pair
                cSim = 1 / (1 + cosine_similarity(x[j], y[j]).mean()) #x[0] cmp w/ y[0-n], x[1] cmp w/ y[0-n], etc.
                dtwSim = 1 / (1 + dtw(x[j],y[j]))
                eucSim = 0 
                for k in range(len(x[j])): 
                    subSim = math.dist(x[j][k], y[j][k]) 
                    eucSim += subSim
                eucSim = 1 / (1 + eucSim / len(x[j])) 
                clsBiSim.append([cSim, dtwSim, eucSim])
                #clsBiSim = [[biMelo cSim, biMelo dtw, biMelo eucSim],
                #            [biRhy cSim , biRhy dtw , biMelo eucSim],
                #            [biPair cSim, biPair dtw, biPair eucSim]]

            # sim for trigram
            for j in range(len(x)): # iteration for different feat: melody, rhythm, pair
                x = i['trigram'] # [n]: n is clmn for N-gram vector:3
                y = work['trigram']
                cSim = 1 / (1 + cosine_similarity(x[j], y[j]).mean()) #x[0] cmp w/ y[0-n], x[1] cmp w/ y[0-n], etc.
                dtwSim = 1 / (1 + dtw(x[j],y[j]))
                eucSim = 0 
                for k in range(len(x[j])):
                    subSim = math.dist(x[j][k], y[j][k]) 
                    eucSim += subSim
                eucSim = 1 / (1 + eucSim / len(x[j])) 
                clsTriSim.append([cSim, dtwSim, eucSim])

        else:
            # sim for bigram
            for j in range(len(x)): # iteration for different feat: melody, rhythm, pair
                cSim = 1 / (1 + cosine_similarity(x[j], y[j]).mean()) #x[0] cmp w/ y[0-n], x[1] cmp w/ y[0-n], etc.
                dtwSim = 1 / (1 + dtw(x[j],y[j]))
                eucSim = 0 
                for k in range(len(y[j])):
                    subSim = math.dist(x[j][k], y[j][k]) 
                    eucSim += subSim
                eucSim = 1 / (1 + eucSim / len(x[j]))             
                clsBiSim.append([cSim, dtwSim, eucSim])

            # sim for trigram
            for j in range(len(x)): # iteration for different feat: melody, rhythm, pair
                x = i['trigram'] # [n]: n is clmn for N-gram vector:3
                y = work['trigram']

                cSim = 1 / (1 + cosine_similarity(x[j], y[j]).mean()) #x[0] cmp w/ y[0-n], x[1] cmp w/ y[0-n], etc.
                dtwSim = 1 / (1 + dtw(x[j],y[j]))
                eucSim = 0 
                for k in range(len(y[j])):
                    subSim = math.dist(x[j][k], y[j][k])  
                    eucSim += subSim
                eucSim = 1 / (1 + eucSim / len(x[j])) 
                clsTriSim.append([cSim, dtwSim, eucSim])

    # res = [composer, [biMelo, biRhy, biPair], [triMelo, triRhy, triPair]]
    # e.g. res[a][b][c] a: composer, b: n-gram, c: cos / dtw / euclidean sim
    return res







os.chdir('maestro-v3.0.0/')
df = pd.read_csv('maestro-v3.0.0.csv')
feat = extract_feat(df)

# select a work
select = feat.iloc[532]
print(select[4], select[5])
res = similarity(select, feat)


# res[a][b][c] a: composer, b: n-gram, c: cos / dtw / euclidean sim
export = []
for a in range(len(res)): # composer iteration
    export.append([res[a][0], 
    res[a][1][0][0], res[a][1][0][1], res[a][1][0][2], # biMelo: cos, dtw, euclidean
    res[a][1][1][0], res[a][1][1][1], res[a][1][1][2], # biRhy: ibid.
    res[a][1][2][0], res[a][1][2][1], res[a][1][2][2], # biPair: ibid.
    
    res[a][2][0][0], res[a][2][0][1], res[a][2][0][2], # triMelo
    res[a][2][1][0], res[a][2][1][1], res[a][2][1][2], # triRhy
    res[a][2][2][0], res[a][2][2][1], res[a][2][2][2]]) # triPair

export = pd.DataFrame(export)
export.to_csv('similarityResult.csv', index=False)