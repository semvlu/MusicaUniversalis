import pandas as pd
import pretty_midi
from mido import MidiFile, MidiTrack
import librosa

import math
import numpy as np
from collections import Counter

from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from tslearn.metrics import dtw

import seaborn as sns
import matplotlib.pyplot as plt 


pd.set_option('display.max_columns', None)
#df = pd.read_csv('maestro-v3.0.0.csv')
#print(df.loc[583])


'''
mid = MidiFile('maestro-v3.0.0/2008/MIDI-Unprocessed_09_R1_2008_01-05_ORIG_MID--AUDIO_09_R1_2008_wav--3.midi')
print(mid)
'''

#-------------------read another file: m1 for comparison-------------------------------------------------------
#
m1 = pretty_midi.PrettyMIDI('lacamp15.midi')
roll1 = m1.get_piano_roll(fs=5) # 60*fs BPM. fs=100, shape = (128, 27515), row=pitch, clmn=time*fs
chroma1 = m1.get_chroma()
roll1 = roll1[21:109]
notes1 = np.argmax(roll1, axis=0) # 1D array
velocity1 = np.max(roll1, axis=0) # ibid. 1D array

notes1 = np.where(notes1 == 0, np.nan, notes1) # replace 0 with nan

# merge as 2D array, [note, velocity], shape = (time*fs, 2)
melody1 = np.stack((notes1, velocity1), axis=-1) 

# get N-gram input, nt_dur = (notes, duration)
# duration unit: fs*durCnt ms, fs=5 pro tem.
nt_dur1 = []
pos = 0
while (pos < len(notes1) - 1):
    durCnt = 1
    while (pos + 1 < len(notes1) and notes1[pos] == notes1[pos + 1] ):
        pos += 1
        durCnt += 1
    nt_dur1.append([notes1[pos], durCnt])
    pos += 1
nt_dur1 = np.array(nt_dur1)
nt_dur1 = nt_dur1[np.all(np.isfinite(nt_dur1), axis=1)] # remove rows w/ nan

# get Unigram------------------------------------------
melodyGram1 = []
rhythmGram1 = []
for i in range(len(nt_dur1)-1):
    gram_note = nt_dur1[i+1][0] - nt_dur1[i][0]
    dur_rt = nt_dur1[i+1][1] / nt_dur1[i][1]
    gram_dur = round(math.log(dur_rt, 2))
    melodyGram1.append(gram_note)
    rhythmGram1.append(gram_dur)
#-----------------------------------------------------

# Construct Bigram
biMelo1 = []
biRhy1 = []
biPair1 = []
biCnt = 0
while (biCnt + 1 < (len(nt_dur1) - 1)):
    biMelo1.append([melodyGram1[biCnt], melodyGram1[biCnt+1]])
    biRhy1.append([rhythmGram1[biCnt], rhythmGram1[biCnt+1]])
    biPair1.append([melodyGram1[biCnt], melodyGram1[biCnt+1], rhythmGram1[biCnt], rhythmGram1[biCnt+1]])
    biCnt += 1
#--------------------------------------------------------------------------------------------------------



# La Campanella by Liszt
m = pretty_midi.PrettyMIDI('maestro-v3.0.0/2008/MIDI-Unprocessed_09_R1_2008_01-05_ORIG_MID--AUDIO_09_R1_2008_wav--3.midi')

roll = m.get_piano_roll(fs=5) # 60*fs BPM. fs=100, shape = (128, 27515), row=pitch, clmn=time*fs
chroma = m.get_chroma()
roll = roll[21:109]
notes = np.argmax(roll, axis=0) # 1D array
velocity = np.max(roll, axis=0) # ibid. 1D array

# fetch melody of piano roll to plot, so each moment has only a note
melody_plt = roll.copy()
for i in range(len(roll[0])): # time*fs
    for j in range(len(roll)): # pitch
        if (melody_plt[j][i] < velocity[i]):
            melody_plt[j][i] = 0

notes = np.where(notes == 0, np.nan, notes) # replace 0 with nan

# merge as 2D array, [note, velocity], shape = (time*fs, 2)
melody = np.stack((notes, velocity), axis=-1) 

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


# get Unigram------------------------------------------
melodyGram = []
rhythmGram = []
for i in range(len(nt_dur)-1):
    gram_note = nt_dur[i+1][0] - nt_dur[i][0]
    dur_rt = nt_dur[i+1][1] / nt_dur[i][1]
    gram_dur = round(math.log(dur_rt, 2))
    melodyGram.append(gram_note)
    rhythmGram.append(gram_dur)
#-----------------------------------------------------

# -----------------Construct Bigram-------------------
biMelo = []
biRhy = []
biPair = []
biCnt = 0
while (biCnt + 1 < (len(nt_dur) - 1)):
    biMelo.append([melodyGram[biCnt], melodyGram[biCnt+1]])
    biRhy.append([rhythmGram[biCnt], rhythmGram[biCnt+1]])
    biPair.append([melodyGram[biCnt], melodyGram[biCnt+1], rhythmGram[biCnt], rhythmGram[biCnt+1]])
    biCnt += 1
#------------------------------------------------------


# -----------------Construct Trigram-------------------
'''
# Construct Trigram
triMelo = []
triRhy = []
triPair = []
triCnt = 0
while (triCnt + 2 < (len(nt_dur) - 1)):
    triMelo.append([melodyGram[triCnt], melodyGram[triCnt+1], melodyGram[triCnt+2]])
    triRhy.append([rhythmGram[triCnt], rhythmGram[triCnt+1], rhythmGram[triCnt+2]])
    triPair.append([ melodyGram[triCnt], melodyGram[triCnt+1], melodyGram[triCnt+2],
    rhythmGram[triCnt], rhythmGram[triCnt+1], rhythmGram[triCnt+2]])

    triCnt += 1
'''
# -----------------------------------------------------


# Zipf's law: x-axis: log(gramCount), y-axis: log(gramRank) is y = -ax + b
counts = Counter(map(tuple, biMelo))
# N/A: it works, but max count = 5 for La Campanella



# Similarity
# for each composer x
# x: other n-gram vectors for composer works, only takes 'train'
# y: N-gram vector to be evaluated, from 'test' & 'validation'
'''
1. Sum up all the similarities, for each composer profile
2. sum.sort(), desc.
3. result = sum.max()
'''


'''
duration = []
pos = 0
while(pos < (len(notes)-1) ):
    durCnt = 0 # duration counter
    posTmp = pos
    for i in range(pos, len(notes)):
        if (notes[i] == notes[pos]):
            durCnt += 1
            posTmp += 1
            if (i == (len(notes)-1)):
                duration.append([notes[pos], durCnt])
                durCnt = 0
                pos = posTmp
                break
        else:
            duration.append([notes[pos], durCnt])
            durCnt = 0
            pos = posTmp
            break
'''



fig, ax = plt.subplots()
'''
ax[0].plot(notes, '.',markersize=1)
ax[1].plot(duration[:,0], '.',markersize=1)
'''
#cgram = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[0])
#fig.colorbar(cgram)

#sns.heatmap(roll)
sns.heatmap(melody_plt)
#ax.plot(notes, '.',markersize=1)
#ax.plot(velocity, '.',markersize=1)

plt.xlabel('Time/5 (s)')
plt.ylabel('Note#')


ax.set_ylim([21,109]) # A0=21, C8=108, piano uses 88 notes

plt.show()
#ax[1].set_ylim([21,108]) 





'''
melodyGram = np.array(melodyGram)


melodyGram1 = np.array(melodyGram1)
#melodyGram1 = melodyGram1.reshape(-1,1)
#melodyGram1 = np.transpose(melodyGram1)

print(melodyGram.shape)
print(melodyGram1.shape)


cSim = 1 / (1 +cosine_similarity(biMelo,biMelo1).mean())
print(cSim)
D, wp = librosa.sequence.dtw(melodyGram, melodyGram1)
sns.heatmap(D)


fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(D, x_axis='frames', y_axis='frames', ax=ax[0])
ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
ax[0].legend()
fig.colorbar(img, ax=ax[0])

ax[1].plot(D[-1,:] / wp.shape[0])
ax[1].set(title='Matching Cost Function')

plt.show()
'''




