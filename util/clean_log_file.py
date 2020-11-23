import os

#log_path = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/Test/IC13/log/log_demo_result.txt'

old_log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrGc_deep/103/log_demo_result.txt', 'r')
#gt_file = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/Test/IC15/gt.txt','w')

new_log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrGc_deep/103/log_cleansing.txt', 'w')


for line in old_log:
    if "/home"  in line:
        new_log.write(line)