import os
import shutil

from PIL import Image

#-----------------------------------------
#last update : 10.6
# save crop img, img's label txt
# make crop image sort
# new upate to non save txt (redundant)
#------------------------------------------

folder_path = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrGc_label/'                    #바운더리파일 과 선이 그어진 기존 파일이 저장되어 있는 디렉토리
imagePath = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrG/'                      # bounadary가 없는 원본이미지 경로
savePath = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrGc/'                     #새 파일들을 저장할 디렉토리

folder_list = os.listdir(folder_path)                    #기존 파일 디렉토리에서 파일 목록 생성
img_list = os.listdir(imagePath)

for folder in folder_list :
    file_list = os.listdir(folder_path+"/"+folder)                    #기존 파일 디렉토리에서 파일 목록 생성
    img_list = os.listdir(imagePath +"/"+folder)

    file_list.sort()
    img_list.sort()


    file_list_txt = [file for file in file_list if file.endswith(".txt")]   #txt 파일만 추려냄

    file_list_jpg = [file for file in img_list if file.endswith(".jpg")]   #jpg 파일만 추려냄
    if len(file_list_jpg)==0:
        file_list_png = [file for file in file_list if file.endswith(".png")]
        png_true = True
    else:
        png_true = False

    def sorting_file(lines):

        newlists = []
        for line in lines:
            line2 = line.split(',')
            line2[6] = line2[6].replace('\n','')
            line3 = list(map(int, line2))
            newlists.append(line3)

        sortedlist= sorted(newlists, key=lambda newlists: newlists[0])
        return sortedlist

    print(folder)
    for i in file_list_txt:                         #텍스트 파일 열기
        f = open(folder_path+"/"+folder+"/"+i,'r')
        lines = f.readlines()
        f.close()
        #
        lines2 = sorting_file(lines)
        #j = i.replace("res_","",1)

        #j = j.replace("txt","jpg")
       # img = Image.open(imagePath + "/" + i.replace("res_", ""))
        if png_true:# png or jpg
            img = Image.open(imagePath +"/"+folder + "/" + i.replace("txt", "png"))
        else:
            img = Image.open(imagePath +"/"+folder + "/" +i.replace("txt", "jpg"))
        fileNum = 1                                 #새로 생성될 파일 넘버링
        for line in lines2:
            #print(line)
            #fileName = i[:-4] + "_" + str(fileNum) + ".txt"
            imageName = i[:-4] + "_" + str(fileNum) + ".png"
            #fw = open(newPath+"/"+fileName, "a")
            #get_list = line.copy()
            #get_list = list(map(str,get_list))
            #line_write = ",".join(get_list)
            #fw.write(line_write)
            #fw.close()


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

            asd = savePath+"/"+folder+"/"+imageName
            if not (os.path.isdir(savePath+"/"+folder)):
                os.makedirs(os.path.join(savePath+"/"+folder))

            #print("{}\t".format(asd))
            #cropping_img = cropping_img.convert()
            cropping_img.save(asd)
            #shutil.copy(path+"/"+i[:-4]+".jpg",newPath+"/"+imageName)   #lines 수 만큼 image파일 생성
            cropping_img.close()
            fileNum = fileNum + 1

