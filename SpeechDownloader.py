"""
File containing scripts to download audio from various datasets

Also has tools to convert audio into numpy
"""
from tqdm import tqdm
import requests
import math
import os
import tarfile
import numpy as np
import librosa
import pandas as pd

import audioUtils


# ##################
# Google Speech Commands Dataset V2
# ##################

# MVPCategs = {'unknown' : 0, 'silence' : 1, '_unknown_' : 0,'_silence_' : 1, '_background_noise_' : 1, 'yes' : 2,
#                 'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 'right' : 7, 'on' : 8, 'off' : 9, 'stop' : 10, 'go' : 11}
# numMVPCategs = 12

# "Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Zero",
# "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", and "Nine"

MVPCategs = {
    'silence': 0,
    'S1': 1,
    'S2': 2,
    'EC': 3,
    'murmur':4
    }
numMVPCategs = 5


def PrepareMVP():
    """
    Prepares Google Speech commands dataset version 2 for used"""
    basePath = '/content/drive/MyDrive/split_e'

    print('Converting test set WAVs to numpy files')
    audioUtils.WAV2Numpy(basePath + '/test/')
    print('Converting test set WAVs to numpy files')
    audioUtils.WAV2Numpy(basePath + '/val/')
    print('Converting training set WAVs to numpy files')
    audioUtils.WAV2Numpy(basePath + '/train/')
    # read split from files and all files in folders
    testWAVs =pd.read_csv('/content/drive/MyDrive/event/testing_list.txt',
                           sep=" ", header=None)[0].tolist()
    valWAVs = pd.read_csv('/content/drive/MyDrive/event/validation_list.txt',
                          sep=" ", header=None)[0].tolist()
    trainWAVs=pd.read_csv('/content/drive/MyDrive/event/train_list.txt',
                          sep=" ", header=None)[0].tolist()
    testWAVs = [f for f in testWAVs ]
    valWAVs = [f for f in valWAVs]
    trainWAVs = [f for f in trainWAVs ]
    # allWAVs = []

    # for root, dirs, files in os.walk(basePath + '/train/'):
    #     allWAVs += [root + '/' + f for f in files if f.endswith('.wav.npy')]
    # trainWAVs = list(set(allWAVs) - set(valWAVs) - set(testWAVs))

    testWAVsREAL = []
    for root, dirs, files in os.walk(basePath + '/test/'):
        testWAVsREAL += [root + '/' +
                         f for f in files if f.endswith('.wav.npy')]

    # get categories
    testWAVlabels = [_getFileCategory(f,MVPCategs) for f in testWAVs]
    valWAVlabels = [_getFileCategory(f, MVPCategs) for f in valWAVs]
    trainWAVlabels = [_getFileCategory(f, MVPCategs) for f in trainWAVs]
    testWAVREALlabels = [_getFileCategory(f, MVPCategs)
                         for f in testWAVsREAL]
    testWAVlabelsDict = dict(zip(testWAVs, testWAVlabels))
    valWAVlabelsDict = dict(zip(valWAVs, valWAVlabels))
    trainWAVlabelsDict = dict(zip(trainWAVs, trainWAVlabels))
    testWAVREALlabelsDict = dict(zip(testWAVsREAL, testWAVREALlabels))

    # a tweak here: we will heavily underuse silence samples because there are few files.
    # we can add them to the training list to reuse them multiple times
    # note that since we already added the files to the label dicts we don't
    # need to do it again

    # for i in range(200):
    #     trainWAVs = trainWAVs + backNoiseFiles

    # info dictionary
    trainInfo = {'files': trainWAVs, 'labels': trainWAVlabelsDict}
    valInfo = {'files': valWAVs, 'labels': valWAVlabelsDict}
    testInfo = {'files': testWAVs, 'labels': testWAVlabelsDict}
    testREALInfo = {'files': testWAVsREAL, 'labels': testWAVREALlabelsDict}
    gscInfo = {'train': trainInfo,
               'test': testInfo,
               'val': valInfo,
               'testREAL': testREALInfo}



    return gscInfo, numMVPCategs

def _getFileCategory(file, catDict):
    """
    Receives a file with name sd_MVP/train/<cat>/<filename> and returns an integer that is catDict[cat]
    """
    categ = os.path.basename(os.path.dirname(file))
    return catDict.get(categ, 0)