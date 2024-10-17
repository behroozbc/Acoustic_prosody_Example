# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 07:50:44 2024

@author: Tomas
"""

import sys
import numpy as np
from scipy.io.wavfile import read

sys.path.append('./Acoustic_prosody')
# from get_phone_features import get_features as get_post
import praat.praat_functions as praat
import sigproc as sg  
import prosody as pr
import mel_extractor as mel
import gamma_extractor as gamma
import matplotlib.pyplot as plt
#**************************************************************************
def filterbank_features(sig,fs,win_time=0.025,step_time=0.01,nfft=1024,N=64,fmax=8000):
    win_size = int(fs*win_time)
    step_size = int(fs*step_time)
    frames = sg.extract_windows(sig,win_size,step_size)
    #Spectrum
    spec_pow,_ = sg.powerspec(frames,fs,win_time,nfft)
    #Mel
    melfb = mel.get_filterbanks(N,nfft,fmax)
    spec_mel = mel.mel_spectrum(spec_pow,melfb)
    mfcc = mel.mfcc_opt(spec_mel,numcep=13)
    
    #Gamma
    gamfb = gamma.generate_filterbank(fs,fmax,win_size,N)
    spec_gam,_ = gamma.cochleagram(spec_pow,gamfb,nfft)
    gfcc = gamma.gfcc(spec_gam)
    
    return np.hstack([mfcc,gfcc])
#****************************************************************************
    
def get_transfeats(sig,fs,f0,label):
    segments,_ = pr.decodef0_transitions(sig,fs,f0,label)
    X = []
    for s in segments:
        flt = filterbank_features(s,fs)
        X.append(flt)
    X = np.vstack(X)
    X = sg.static_feats(X)
    return X
#****************************************************************************


#Phonation: Requires sustained vowel production: aaaaaaaaaaaa...
#Pitch
fs,sig = read('vowel.wav')
sig = sig-np.mean(sig)
sig = sig/np.max(np.absolute(sig))
f0 = pr.f0_contour_pr(sig,fs)
uf0 = np.mean(f0[f0>0])
sf0 = np.std(f0[f0>0])
#Loudness
spl = pr.sound_pressure_level(sig,fs)
uspl = np.mean(spl[spl>0])
sspl = np.std(spl[spl>0])
#Perturbation
jit = pr.ppq(f0,2)
ppq3 = pr.ppq(f0,3)
ppq5 = pr.ppq(f0,5)
shimm = pr.apq(spl,2)
apq3 = pr.apq(spl,3)
apq5 = pr.apq(spl,5)   
arr=[]

#Prosody: Text, sentences, long speech.
fs,sig = read('audio.wav')
sig = sig-np.mean(sig)
sig = sig/np.max(np.absolute(sig))
X_pro = pr.prosody_features(sig,fs)
X_values = np.asarray(list(X_pro.values()))
arr.append(X_pro)

#Articulation
#Compute articulation features
f0 = pr.f0_contour_pr(sig,fs)
Xon = get_transfeats(sig,fs,f0,'onset')
Xoff = get_transfeats(sig,fs,f0,'offset')
X_Art = np.hstack([Xon,Xoff])

