import collections
from PIL import Image
import os
import shutil



'''
# # copy image 
# # generate answer txt file same name.
'''


log_path = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/Test/IC13/log/log_gt.txt'
file_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeB/'

file = open(log_path, mode='rt')
isinstance(file, collections.Iterable)

total_log_file = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeB_label.txt', 'w')

count =0

print("start copy with log")

for line in file:
    #print("[",line,"]")
    if(line=="\n"):
        break;

    if("/home" in line):


        dir = line.split("\t")[0]
        answer = line.split("\t")[2]
        deep_text_predict_label = answer.strip()
        file_name = dir.split("/")[10]
        original_label = file_name.split("_")[1]

        if deep_text_predict_label != original_label:
            count += 1
            total_log_file.write("{}\t{}\n".format(file_dir+"/IC13/"+file_name, deep_text_predict_label))

            shutil.copy(dir, file_dir+"/IC13/")









total_log_file.write("file : {}\n".format(count))
file.close()
total_log_file.close()
print(count)
print("Done!")