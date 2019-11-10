import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
audios = 'cut_audio_files/'
pictures = 'raw_pictures/'
audio_fpath = "/home/hozaifa/comp_lit/final-project/"
audio_clips = os.listdir(audio_fpath+audios)
def crop_img_func(audio_fpath, pictures, audio_clip_num):
	im = cv2.imread(audio_fpath + pictures+"foo" + str(audio_clip_num) +".png",1)
	#print (type(im)) #Print <class 'numpy.ndarray'>
	#print (im.shape) #prints 2100000
	446
	im = im[61:446, 4000:28800]
	#img = Image.fromarray(im, 'RGB')
	#img.show()
	return im

