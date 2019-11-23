import numpy as np
import random
def CreateBatch(batch_size, multiplier, full_script, Phonemes, n_classes, batch= None):
	fake_batch_x = []
	fake_batch_y = []
	for training_ex in range(batch_size):
		z = random.randint(1,156000)

		#75% chance that the training example comes from the new batch introduced
		if random.randint(1,4) >=2 and multiplier>1:
			while not z > ((6000*multiplier)-5950) or not z< ((6000*multiplier)-50) or z%6000 < 50 or z%6000 >5970 or full_script[z-1][0] == 'oov' or full_script[z-1][0] == 'not-found-in-audio' or full_script[z-1][0] == 'silence':
				z = random.randint(0,156000)
			#print ('first for loop')
			#print (z)
		#Else it will pick one from before the new batch, therefore it can't forget old stuff
		elif multiplier>1:
			while not z > 100 or not z< ((6000*multiplier)-6000) or  z%6000 < 50 or z%6000 >5970 or  full_script[z-1][0] == 'oov' or full_script[z-1][0] == 'not-found-in-audio' or full_script[z-1][0] == 'silence':
				z = random.randint(0,156000)
			#print ('second for loop')
			#print (z)
		else:
			while not z > ((6000*multiplier)-5950) or not z< ((6000*multiplier)-50) or full_script[z-1][0] == 'oov' or full_script[z-1][0] == 'not-found-in-audio' or full_script[z-1][0] == 'silence':
				z = random.randint(0,156000)
			#print ('third for loop')
			#print (z)
		img = np.load('all_spectograms/img_'+str(z)+'.npy')/255

		if training_ex == 0:
			batch_x = np.reshape(img, (1,385,165,3))
		else:
			batch_x = np.concatenate((batch_x, np.reshape(img, (1,385,165,3))),0)
		if full_script[z-1][0][-2] == '_':
			fake_batch_y.append(Phonemes[full_script[z-1][0][:-2]])
		else:
			fake_batch_y.append(Phonemes[full_script[z-1][0]])
			#print ('using frame ' + str(z) + ' and the label is ' + str(fake_batch_y[-1]))

	print ('created batch_x with a shape of ' + str(batch_x.shape)+ ' and created fake_batch_y with a length of ' + str(len(fake_batch_y)) + ' for batch ' + str(batch))

	#batch_x is defined as a numpy array of fake_batch_x		

	assistant_y = np.zeros((batch_size,n_classes))
	fake_batch_y = np.array(fake_batch_y)
	#print (np.arange(batch_size))
	#print (fake_batch_y)
	batch_y = assistant_y[np.arange(batch_size),fake_batch_y] = 1
	batch_y = assistant_y
	print ('created batch y with a shape of ' + str(batch_y.shape))

	return (batch_x,batch_y)
	#print (batch_y)
