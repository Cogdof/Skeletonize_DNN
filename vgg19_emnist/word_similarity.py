import os
import shutil

from PIL import Image
import Levenshtein
'''
#-----------------------------------------
20.09.28

Compare pred score and gt.
Get Similarity score to use at performance at connected network


#------------------------------------------
'''
gt_path = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/dataset/image/[IC13]Challenge2_Test'                  #기존 파일이 저장되어 있는 디렉토리
log_path = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/'


result_log = open(log_path+"/log_demo_result.txt",'r')
distance_log = open(log_path+"/log_demo_result_distance.txt",'a')
gt_file = open(gt_path+"/gt.txt", 'r').readlines()



include_str = "png"


print("n" in include_str)
def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
j=0
for i in result_log:

    if include_str in i and j < len(gt_file):
        pred = i.split("\t")[1]
        gt = gt_file[j].split("\t")[1]
        print(pred, "\t ",gt)
       #distance = edit_distance(pred, gt) / max([len(pred), len(gt)])




        sim = Levenshtein.distance(pred,gt)
        print(i.replace("\n",""),"\t",sim )

        distance_log.write(i.replace("\n","")+ "\t")
        distance_log.write(str(sim))
        distance_log.write("\n")

        j = j+1





distance_log.close()
result_log.close()


'''

path = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/data'                  #기존 파일이 저장되어 있는 디렉토리

file_list = os.listdir(path)


for i in file_list:                         #텍스트 파일 열기


    gt_file = open(path+"/{}/gt.txt".format(i), 'a')
    img_file_list = os.listdir(path+"/{}/data/".format(i))
    for j in img_file_list:
        arr = j.split("_")
        label = arr[1]
        gt_file.write("data/{}\t{}\n".format(j, label))




    gt_file.close()

'''