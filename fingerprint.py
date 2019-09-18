import os
import librosa
from sklearn import mixture
from sklearn import svm
import numpy as np
from tqdm import tqdm
from collections import Counter
import operator
from sklearn.ensemble import IsolationForest
import pickle
import matplotlib.pyplot as plt
from sklearn import manifold
import hashlib
import numpy as np
from scipy import signal, fft, arange
import librosa.display
import sklearn.preprocessing

def getFeatures(path):
    albumsDir =[]
    genreDirectory = os.getcwd()+ path

    #go through discography and generate a list of dirs of albums

    titles = []
    
    for root, dirs, files in os.walk(genreDirectory):
        for file in files:
            if file[0] != '.':
                titles.append(genreDirectory+"/"+file)

    spectograms = []
    
    for title in tqdm(titles):
        y, sr = librosa.load(title)
        # f, t, spec = signal.spectrogram(y, sr)
        spec = librosa.feature.melspectrogram( y=y, sr=sr)
        spectograms.append(spec)
    
    
    return spectograms, titles


def spec2hash(spec):
    
    final_hash = []
    spec_transposed = spec.transpose()
    # y axis: freq
    # x axis: time window
    x, y = spec_transposed.shape
    #print(x, y)

    #normalizing input
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    spec_normalized = min_max_scaler.fit_transform(spec_transposed)
    
    
    output = np.dot(spec_normalized,layer1) # output shape = time x 200
    
    for cell in np.nditer(output, op_flags=['readwrite']):
        if cell >= 0.5:
            cell[...] =1
        else:
            cell[...] =-1
    output = output.tolist()
    hash_keys = [output[i:i+5] for i in range(len(output)-5)]
    flattened_hash_keys =[np.asarray(key).flatten().tolist() for key in hash_keys]
        
    return flattened_hash_keys



##defining matrixes (NN layers)
layer1 = np.random.randn(128,32)


####PSEUDOCODE
# x = load som
# X = espectrograma (x)
# A = matriz_rand
# ids = sinal(X * A)
# hashes = [hash(ids[i:i+5]) for i in range(len(A)-5)]

spec, titles = getFeatures('/TZ_Dataset')
# GENERATING DICTIONARY (Dataset)

removed_keys = []

single_music_hash = {}
for i in tqdm(range(len(spec))):
    key_list = spec2hash(spec[i])
    key_list = [tuple(element) for element in key_list]
    
    for element in key_list:
        #if it has not been found before (duplicate windows in tracks (begining silence))
        if element not in removed_keys:
            if element not in single_music_hash:
                single_music_hash[element] = titles[i]
            else: 
                removed_keys.append(element)
                single_music_hash.pop(element)


# print("------------\nDictionary created\n------------")


features_to_find, titles_to_find = getFeatures('/TZ_Test')
test_music_hash = []
for feature in tqdm(features_to_find):
     test_music_hash.append(spec2hash(feature))

# print("------------\nTest List created\n------------")

        
for i in range(len(test_music_hash)):
    for element in test_music_hash[i]:
        if tuple(element) in single_music_hash:
            print (titles_to_find[i].split('/')[-1], " -> ",single_music_hash[tuple(element)].split('/')[-1])
            # testing for the same string (positive result)
            break
	# print(titles_to_find[i], " not found on the set provided ")	
