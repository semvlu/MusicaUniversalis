import os
import pandas as pd
import numpy as np
from operator import attrgetter
import matplotlib.pyplot as plt 

os.chdir('maestro-v3.0.0/')
df = pd.read_csv('maestro-v3.0.0.csv')

valid = df.index[df['split'] =='validation'].tolist()
test = df.index[df['split'] == 'test'].tolist()

print(valid)
print(test)



c = df.canonical_composer.value_counts()
#print(df[df['canonical_composer'] == 'Henry Purcell'])
#print(df[df.canonical_composer.isin(c.index[c.le(60)])]) #composers that has only 1 work is @ train



fig, ax = plt.subplots()
#ax.pie(c,labels=c.index, autopct='%1.0f%%')
plt.subplots_adjust(bottom=0.25)
ax.plot(c,'o-')
for i, v in enumerate(c):
    ax.annotate(str(c[i]), xy=(i,v), xytext=(-7,7), textcoords='offset points')
plt.xticks(rotation=45, ha='right', fontsize=8)

ax.set_title('Works Distrinution by Composer')
plt.show()



'''
for i in df['canonical_composer']:
    print(i)
'''



# 60 composers
# composer w/ only 1 work,
# work w/ multi composers
seen = set()
for row in df['canonical_composer']:
    if row in seen:
        continue
    else:
        seen.add(row)
        print(row)
print('Total #composer: ', len(seen))
# 54 composers after removal

'''
mul = 'Sergei Rachmaninoff / György Cziffra'
if('/' in mul):
    mul1 = mul.split('/') # mul = ['Sergei Rachmaninoff ', ' György Cziffra']
    print(mul1)
    mul1[0] = mul1[0][:-1]
    mul1[1] = mul1[1][1:]
    print(mul1)
'''


