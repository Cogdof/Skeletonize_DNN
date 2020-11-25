import os
import shutil


'''
# read log, move file

'''


befor_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc/IC13/'
after_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc_notDeep/'


log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc/TeBc_gt/save_log.txt','r')

dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc/IC13'
list = []
for i in log:
    if "IC13" in i:
        file_name = i.replace("IC13/","").split("\t")[0]
        shutil.copy(dir+"/"+file_name , after_dir + "/IC13_"+file_name)

