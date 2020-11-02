from PIL import Image
import os


folder_dir ="/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/Fnt (사본)/"

folder_list = os.listdir(folder_dir)


for i in folder_list:

    file_dir = folder_dir+i
    file_list = os.listdir(file_dir)
    print(i)
    for j in  file_list:
        file = file_dir+"/"+j


        img = Image.open(file)
        img = img.convert("RGBA")
        datas = img.getdata()

        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                if item[0] > 150:
                    newData.append((0, 0, 0, 255))
                else:
                    newData.append(item)



        img.putdata(newData)
        img.save(file, "PNG")
print("Done!")