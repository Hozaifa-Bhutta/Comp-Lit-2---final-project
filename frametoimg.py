from crop_img import crop_img_func
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np
audios = 'cut_audio_files/'
pictures = 'raw_pictures/'
audio_fpath = "/home/hozaifa/comp_lit/final-project/"
audio_clips = os.listdir(audio_fpath+audios)

bottom_border = 455
one_second_length = 413.333333
width = 24800
height = 446
depth = 3

def framenumToimg(frame_num):
	#Determines the second number based on frame number
	second = (frame_num/100)
	audio_clip_number = int(second/60)
	second = second-60*audio_clip_number
	second_fraction = math.modf(second)[0]
	whole_second = int(second)
	exact_pixel = one_second_length*second
	minus_50 = exact_pixel-(0.5*one_second_length)
	plus_50 = exact_pixel+(0.50*one_second_length)
	im = crop_img_func(audio_fpath,pictures,audio_clip_number)
	im = im[:, int(minus_50):int(plus_50)]
	if im.shape[1] == 414:
		im = im[:, :-1]

	#print (type(im)) #Print <class 'numpy.ndarray'>
	#print (im.size) #prints 619500
	#print (im.shape)#(385, 413, 3)
	return im
	#img = Image.fromarray(im, 'RGB')
	#img.show()	

