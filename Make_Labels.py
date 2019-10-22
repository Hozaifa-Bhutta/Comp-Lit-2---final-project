def makearray(file):
    import numpy as np
    import json
    f = open(file)
    #make the json file a dictionary
    dict = json.load(f)
    array = dict["words"]
    print(len(array))
    return array


def makelabel(file, framerate):
    import numpy as np
    import json
    f = open(file)
    #make the json file a dictionary
    dict = json.load(f)
    array = dict["words"]
    frame_phonemes = []
    #print(len(array))
    #we define a buffer to account for all the "silent" phonemes at the beginning
    #buffer = ['silence']
    #buffer *= int(array[1]['start']*30)
    #frame_phonemes += buffer
    #print(frame_phonemes)
    #we start a series of for loops that keeps adding phonemes to the final list
    prev_word = 0
    for word in array:
        #print(word)
        print(frame_phonemes)
        phone_list = []
        #change this to make more efficient
        try:
            phone_list += ['silence'] * int(framerate * (word['start'] - prev_word))
            for phoneme in word['phones']:
                #add the phoneme to a temporary list and multiply it to get the number of frames right
                phone = []
                #phone_frames = []
                phone.append(phoneme['phone'])
                #phone_frames += ['silence'] * int(30 * (word['start'] - prev_word))
                phone *= int(framerate * float(phoneme['duration']))
                phone_list += phone
            frame_phonemes += phone_list
            prev_word = word['end']
        except:
            print('fail')

    return frame_phonemes
