import collections
from PIL import Image
import os

# copy image with recognition over setting threshold (60%)
# generate answer txt file same name.


log_path = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/log_high.txt'
file_dir = '/home/mll/v_mll3/OCR_data/인식_100데이터셋/word단위_with_label/result_model_mlt25k_ic15_recogtitiontrainset_new/'

file = open(log_path, mode='rt')
isinstance(file, collections.Iterable)

for line in file:
    #print("[",line,"]")
    if(line=="\n"):
        break;
    str = line.split("\t")[0]
    answer = line.split("\t")[1]
    print(str)
    #print(answer)
    str2 = str.split("/")
    str_rename = str2[6]+"_"+str2[7]

    # dir generate
    target_dir = file_dir + str2[6]+'/'
    if not (os.path.isdir(target_dir)):  # 새  파일들을 저장할 디렉토리를 생성
        os.makedirs(os.path.join(target_dir))

    #print(str_rename)
    img_file = Image.open(str)
    #img_file.show()
    img_file.save(target_dir + str_rename)



    #text answer
    str_answer = str2[6]+"_"+str2[7].split(".")[0]+".txt"
    #print(str_answer)
    answer_file = open(target_dir+str_answer,'w')
    answer_file.write(answer)
    #answer_file.write('\n')
    #answer_file.write(len(answer))
    answer_file.close()
    #print(str)

file.close()