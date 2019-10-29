from scipy import signal
from scipy.io import wavfile
import pandas as pd
import os
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

#for loading and visualizing audio files
import librosa
import librosa.display

#to play audio
import IPython.display as ipd

audio_fpath = "/home/hozaifa/comp_lit/videos/audios/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",(audio_clips))
x, sr = librosa.load(audio_fpath+audio_clips[1], sr=44100)

print(type(x), type(sr))
print(x.shape, sr)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.show()
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

plt.colorbar()
plt.show()
