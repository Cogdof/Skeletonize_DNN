import os
import shutil

from PIL import Image

path = '/home/mll/v_mll3/OCR_data/result_set/craft_result/charater_seperate_recognition_result/Challenge2_Training_Task12_result'                  #기존 파일이 저장되어 있는 디렉토리
newPath = '/home/mll/v_mll3/OCR_data/result_set/seperate by character/Challenge2_Training_Task12'           #새 파일들을 저장할 디렉토리

file_list = os.listdir(path)                    #기존 파일 디렉토리에서 파일 목록 생성

if not(os.path.isdir(newPath)):                 #새  파일들을 저장할 디렉토리를 생성
    os.makedirs(os.path.join(newPath))

file_list_txt = [file for file in file_list if file.endswith(".txt")]   #txt 파일만 추려냄
file_list_jpg = [file for file in file_list if file.endswith(".jpg")]   #jpg 파일만 추려냄


for i in file_list_txt:                         #텍스트 파일 열기
    f = open(path+"/"+i,'r')
    lines = f.readlines()
    f.close()
    img = Image.open(path+"/"+i.replace("txt", "jpg"))
    fileNum = 1                                 #새로 생성될 파일 넘버링

    for line in lines:
        #print(line)
        fileName = i[:-4] + "_" + str(fileNum) + ".txt"
        imageName = i[:-4] + "_" + str(fileNum) + ".jpg"
        fw = open(newPath+"/"+fileName, "a")
        fw.write(line)
        fw.close()

        line = line.split(',')
        #line = line.split('\t')
        #line.remove(line[0])
        line[6] = line[6].replace('\n','')
        #line[7] = line[7].replace('\n', '')

        line = list(map(int, line))
        #if (int(lines[2])):
        #    area = (int(lines[1]), int(lines[2]), int(lines[5]), int(lines[6]))
        #else :
        line_x = []
        line_y = []
        line_x.append(line[0])
        line_x.append(line[2])
        line_x.append(line[4])
        line_x.append(line[6])

        line_y.append(line[1])
        line_y.append(line[3])
        line_y.append(line[5])
        line_y.append(line[7])
        area = (min(line_x), min(line_y), max(line_x), max(line_y))
       # area = (int(lines[1]), int(lines[2]), int(lines[5]), int(lines[6]))

        cropping_img = img.crop(area)

        asd = newPath+"/"+imageName
        print(asd)
        cropping_img.save(asd)
        #shutil.copy(path+"/"+i[:-4]+".jpg",newPath+"/"+imageName)   #lines 수 만큼 image파일 생성
        cropping_img.close()
        fileNum = fileNum + 1

