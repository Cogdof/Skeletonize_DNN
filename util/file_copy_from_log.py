import collections
from PIL import Image
import os
import shutil



'''
# # copy image 
# # generate answer txt file same name.
'''


log_path = '/home/mll/v_mll3/OCR_data/final_dataset/MJ_set log/log_demo_result.txt'
file_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrG/'

file = open(log_path, mode='rt')
isinstance(file, collections.Iterable)

total_log_file = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrG_label.txt', 'w')

count =0
folder = 0

print("start copy with log")
folder_log_file = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrG_log/{}_TrG_label.txt'.format({folder}), 'w')
for line in file:
    #print("[",line,"]")
    if(line=="\n"):
        break;

    if("/home" in line):


        dir = line.split("\t")[0]
        answer = line.split("\t")[1]
        deep_text_predict_label = answer.strip()
        file_name = dir.split("/")[13]
        original_label = file_name.split("_")[1]

        if deep_text_predict_label == original_label:

            if count % 10000 == 0:
                folder += 1
                print(folder)
                folder_log_file.close()
                folder_log_file = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrG_log/{}_TrG_label.txt'.format(folder), 'w')
                if not (os.path.isdir(file_dir + str(folder))):  # 새  파일들을 저장할 디렉토리를 생성
                    os.makedirs(os.path.join(file_dir + str(folder)))
                new_file_dir = file_dir + str(folder) + "/"

            count += 1
            total_log_file.write("{}{}\n".format(new_file_dir,file_name))
            folder_log_file.write("{}{}\n".format(new_file_dir,file_name))
            shutil.copy(dir, new_file_dir + '/')









total_log_file.write("file : {}\n".format(count))
file.close()
total_log_file.close()
print(count)
print("Done!")