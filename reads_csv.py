#reads_csv.py
import csv
import numpy as np
def read_file(file_name, line_count_p):
	with open(file_name) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		'''Turns first image into a numpy array'''
		for row in csv_reader:
			if line_count == line_count_p:
				x = np.array(row)
				break
			else:
				if line_count%30==0:
					print ('new line')
					
				line_count+=1
		return x  
