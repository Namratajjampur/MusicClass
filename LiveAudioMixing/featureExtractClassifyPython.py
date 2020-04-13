#!/usr/bin/env python
# coding: utf-8

# In[43]:


url=r"D:\mastered\B.wav"
audio_path = url
x, sr = librosa.load(url)
librosa.display.waveplot(x, sr=sr)


# In[44]:


url=r"D:\unmastered\B.wav"
audio_path = url
x, sr = librosa.load(url)
librosa.display.waveplot(x, sr=sr)


# In[11]:


mfccs = librosa.feature.mfcc(x, sr=sr,n_mfcc=20)
mfccs.shape


# In[45]:


import numpy as np
mfccsscaled = np.mean(mfccs.T,axis=0)
len(list(mfccsscaled))


# In[55]:


import librosa,librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import librosa
import glob 
import sklearn
import os
import csv
import IPython.display as ipd
#ipd.Audio('/Users/niranjandr/Music/Music/224031__akshaylaya__bheem-b-002.wav')


# In[57]:


#creating a csv file with file description and respective classes

path = r'C:\Users\rames\OneDrive\Desktop\namma\MusicLiveAudioMixing\sounds'

with open('metadata.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['folder', 'file', 'class'])
    for root, dirs, files in os.walk(path):
        #print(root,"abc",dirs,"abc",root.split(path))
        for filename in files:
            #print(root,dirs,files)
            a=0
            if(root.split(path)[1][1:]=='mridanga'):
                a=1
            elif(root.split(path)[1][1:]=='violin'):
                a=2
            elif(root.split(path)[1][1:]=='vocals'):
                a=3
            writer.writerow([root.split(path)[1][1:], os.path.join(root,filename),a])


# In[35]:


#reading from csv file, extracting MFCCs and creating a new dataframe


# In[58]:


#trying for first 5 
df=pd.read_csv("metadata.csv")


# In[59]:


mfccs_all=[]
for i in range(len(df)) : 
    #print(df.iloc[i, 1]) 
    #ipd.Audio(df.iloc[i,1])
    x, sr = librosa.load(df.iloc[i,1])
    #Plot the signal:
    #plt.figure(figsize=(14, 5))
    #librosa.display.waveplot(x, sr=sr)
    #extracting 20 mfccs
    mfccs = librosa.feature.mfcc(x, sr=sr,n_mfcc=20)
    #scaling to 1D
    mfccsscaled = list(np.mean(mfccs.T,axis=0))
    mfccs_all.append(mfccsscaled)
    


# In[60]:


df_features = pd.DataFrame(mfccs_all) 


# In[61]:


#concatenating the 2 datasets
result = pd.concat([df, df_features], axis=1, sort=False)


# In[62]:


result


# In[63]:


result.to_csv('features_phase1.csv') 


# In[14]:


import sklearn as sk
from sklearn import svm
import pandas as pd


# In[16]:


result = pd.read_csv("features_phase1.csv") 
X = result.drop(['folder','file','class'], axis=1)
y = result['class']


# In[72]:


#y


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[19]:


#X_train


# In[69]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


# In[70]:


y_pred = svclassifier.predict(X_test)


# In[71]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[8]:


import pickle
import librosa
import numpy as np


# In[9]:


filename = 'finalized_classification_model.sav'
#pickle.dump(svclassifier, open(filename, 'wb'))


# In[20]:


loaded_model = pickle.load(open(filename, 'rb'))
x, sr = librosa.load(r'D:\Unmastered\Vocal.wav')
mfccs = librosa.feature.mfcc(x, sr=sr,n_mfcc=20)
#scaling to 1D
mfccsscaled = list(np.mean(mfccs.T,axis=0))


# In[25]:


pd.DataFrame([mfccsscaled])


# In[27]:


result = loaded_model.predict(pd.DataFrame([mfccsscaled]))
print(result)


# In[ ]:




