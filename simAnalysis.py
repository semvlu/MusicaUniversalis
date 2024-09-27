import pandas as pd
import os

os.chdir('maestro-v3.0.0/')
df = pd.read_csv('Franz Schubert,Son. in Bb.csv', names=
['Composer', 
'Melody bigram cos', 'Melody bigram dtw', 'melody bigram euclidean', 
'Rhythm bigram cos', 'Rhythm bigram dtw', 'Rhythm bigram euclidean', 
'Combined bigram cos', 'Combined bigram dtw', 'Combined bigram euclidean',
'Melody trigram cos', 'Melody trigram dtw', 'melody trigram euclidean', 
'Rhythm trigram cos', 'Rhythm trigram dtw', 'Rhythm trigram euclidean', 
'Combined trigram cos', 'Combined trigram dtw', 'Combined trigram euclidean'])
df = df.iloc[1:] 
# closer to 1, more similar
# calculate distance to 1 by abs, now closer to 0, more similar
df.iloc[:, 1:] = abs((df.iloc[:, 1:] - 1))


min_val= df.min()
min_idx = df.idxmin()
for i in range(1, len(min_idx)):
    print(df['Composer'][min_idx[i]], min_val.iloc[i])
