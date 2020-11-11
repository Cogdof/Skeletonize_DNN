import os
import shutil
from string import ascii_letters
'''

classify single character synth data.

to each folder.


* condition score

1. confidence score 60% over
2. Only detected one label of alphabet. other is wrong detected.


'''


log_dir = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/result_log/개정_craft_seperate/log_demo_result.txt'


after_dir = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/dataset/seperate_good/'




log_file = open(log_dir,"r")

list = list(ascii_letters)
for i in range(0,10):
    list.append(str(i))

for a in list:
    if not (os.path.isdir(after_dir + a)):
        os.makedirs(os.path.join(after_dir + a))

count = 0

for line in log_file:
    if "/home" in line:
        label = line.split("\t")[1]
        label = label.replace(" ","")
        folder = line.split("/")[8]
        image_name =  (line.split("/")[9]).split("\t")[0]
        score = float(line.split("\t")[2])
        image_dir = line.split("\t")[0]

        if len(label) ==1 and score > 0.8:
            #print(label)
            if os.path.isdir(after_dir+label):
                shutil.copy(image_dir, after_dir + label+"/"+folder+"_"+image_name)
            else:
                shutil.copy(image_dir, after_dir + "etc/")

            count = count+1

print("move over : ",count)

