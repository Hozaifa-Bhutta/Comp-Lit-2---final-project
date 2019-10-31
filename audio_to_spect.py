from PIL import Image
import librosa
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
#for loading and visualizing audio files
import librosa
import librosa.display
import pylab
#to play audio
import sys
np.set_printoptions(threshold=sys.maxsize)

pylab.axis('off') # no axis
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
audios = 'cut_audio_files/'
pictures = 'raw_pictures/'
audio_fpath = "/Users/Hozai/Desktop/audios/"
audio_clips = os.listdir(audio_fpath+audios)
print("No. of .wav files in audio folder = ",(audio_clips))
x, sr = librosa.load(audio_fpath+audios+audio_clips[0], sr=44100)

print(type(x), type(sr))
print(x.shape, sr)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.savefig(audio_fpath +pictures+'foo.png')

plt.colorbar()
plt.show()

#Opens the image
	#im = Image.open(audio_fpath + 'foo.png', 'r')
#Saves it as numpy array
	#na = np.array(im)

#Saves it in npy 
	#np.save('test3.npy', na)


#Loads it back
	#d = np.load('test3.npy')


im = cv2.imread(audio_fpath + pictures+"foo.png",1)
print (type(im)) #Print <class 'numpy.ndarray'>
print (im.size) #prints 2100000
img = Image.fromarray(im, 'RGB')
img.show()
