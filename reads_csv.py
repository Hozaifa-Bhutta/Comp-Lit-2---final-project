#reads_csv.py
import csv
import numpy as np

with open('banana_pics.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    '''Turns first image into a numpy array'''
    for row in csv_reader:
        if line_count == 0:
            print (type(row))
            x = np.array(row)
            print (x)
            line_count+=1
        else:
             break       