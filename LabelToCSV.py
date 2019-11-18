
#At one point, two words are connected as there is no silence in between them
for i in range( 1, 27):
	framerate = 100
	import numpy as np
	import json
	import math
	f = open("/Users/Hozai/Desktop/JSON_files/min_" + str(i) + ".json")
	#make the json file a dictionary
	dict = json.load(f)
	array = dict["words"]
	frame_phonemes = []
	#we start a series of for loops that keeps adding phonemes to the final list
	prev_word = 0
	error = False
	for word in array:
		#print(word)
		#print(frame_phonemes)
		word_list = []
		#change this to make more efficient
		try:
			if error == True:
				word_list += ['not-found-in-audio'] * int(round(framerate * (round(word['start'],2) - prev_word),0)-1)
				error = False
			else:
				word_list += ['silence'] * int(round(framerate * (round(word['start'],2) - prev_word),0) -1)
				#print (len(word_list))
			first_phone = True
			counter = 1
			#print (word[])
			#print (word['alignedWord'])

			

			for phoneme in word['phones']:
				#add the phoneme to a temporary list and multiply it to get the number of frames right
				phone = []
				if word['alignedWord'] == 'There' and  word['end'] == '54.319998':
					print (len(phoneme))

				#phone_frames = []
				phone.append(phoneme['phone'])
				#phone_frames += ['silence'] * int(30 * (word['start'] - prev_word))
				if counter == len(word['phones']):
					phone *= int(round(framerate * round(float(phoneme['duration']),2),0) + 1)
					#print (len(phone))

							#print("yeeeeeeeeeet")
				else:
					phone *= int(framerate * round(float(phoneme['duration']),2))
					'''if word['alignedWord'] == 'bikini':
						print (word['phones'])
						print (phone[0] + ' ' + str(len(phone)) + ' ' (phoneme['duration']))'''
					#print (phoneme['duration'])
					#print (len(phone))

				if word['start'] == prev_word and counter == 1:
					print ('okay')
					print (word)
					print (word_list)
					if len(phone) != 1:
						del phone[0] 
					else:
						del frame_phonemes[-1]

				word_list += phone
				counter += 1
			#If a problem occurs due to rounding and an extra frame is produced, this turns the first frame of the word into silence. 
			#However, this will not be implemented and instead done manually as the first 'frame' might contain more important information than the last 'frame' of the previous word


			frame_phonemes += word_list
			prev_word = round(word['end'], 2)
			first_phone = False
		except Exception as e:
			print('fail')
			print (e)
			error = True
			print (word)
	with open("CSV_files/min_" + str(i) + ".csv", "w") as myfile:
		for entries in frame_phonemes:
			myfile.write(entries)
			myfile.write("\n")
	
