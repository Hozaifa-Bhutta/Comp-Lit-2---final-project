from crop_img import crop_img_func
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np
filename = 'audios/test'
audios = 'cut_audio_files/'
pictures = 'raw_pictures/'
audio_fpath = "/Users/Hozai/Desktop/audios/"
audio_clips = os.listdir(audio_fpath+audios)

bottom_border = 455
def framenumToimg(frame_num):
	#Determines the second number based on frame number
	second = (frame_num/100)
	second_fraction = math.modf(second)[0]
	if frame_num > 51:
		raw_im1 = crop_img_func(audio_fpath,pictures,0)
		raw_im2 = crop_img_func(audio_fpath,pictures,1)
		#im = raw_im1[:bottom_border, second*len:length[1]]


		height = raw_im1.shape[0]
		width = raw_im1.shape[1]

		main_frame_pix = (second_fraction*width)

		im1 = raw_im1[:, int((main_frame_pix)-(0.5*width)):]

		remaining_frames = 100 - frame_num
		im2 = raw_im2[:, :int((remaining_frames/100)*width)]
		print (im2.shape)
		
		#deletes borders


		img1 = Image.fromarray(im1, 'RGB')

		img2 = Image.fromarray(im2, 'RGB')

		final_im = np.concatenate((im1,im2), axis=1)
		final_img = Image.fromarray(final_im, 'RGB')
		final_img.show()
	

	
framenumToimg(60)
