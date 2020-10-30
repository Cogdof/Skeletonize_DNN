import cv2
import os
import PIL.Image as IMG
import numpy as np
'''
------------------------------------
# [20.10.23 build]
# Generate synth char data.

merge wild background img and char font img.

img type : RGB, HSB, YUB ?

label :
ver1 : 47 0-9 a-z + @ (ABDEFGHNQRT)
ver2 : 62 0-9 a-z A-Z

[Lastest update] : 2020.10.27 

------------------------------------
'''
file_dir  = "/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/v1/"

def load_font(label):
    #load font style with label

    font_dir = "/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/Fnt/"

    #open label[x] 's dir
    font_dir = font_dir+"/"+label
    font_list = os.listdir(font_dir)
    fontimg_list = []


    for i in font_list:
        im = IMG.open(font_dir+"/"+i)
        nparry_im = np.array(im)
        fontimg_list.append(nparry_im)
        #fontimg_list.append(img_convert(i))
    # img convert
    return fontimg_list


def img_convert(font_list):
    # Add random effect to img like noise, Change color, opacity, blur

    return 0

def load_background():
    # Get any wild image, -> split size AxB

    backimg_dir = "/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/English_폰트/Fnt/"

    return 0

def belnding(label):
    # merge font and background

    # opaci
    a =0.5
    b =0.5

    fonts = load_font(label)
    background = load_background()

    name = 0
    for i in fonts:
        for j in background:

            j = cv2.resize(i.shape[1], i.shape[0])  #same size with two img
            name +=1
            new_img = cv2.addWeighted(i, a, j, b, 0)
            cv2.imshow(new_img)
            cv2.imwrite(file_dir+"/"+label+"/"+label+"_"+str(name)+".png", new_img)




def main():

    #file_dir



    label47 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    # 47 label

    ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    digits = '0123456789'
    label62 = []
    for i in digits+ascii_uppercase+ascii_lowercase:
        label62.append(i)

    #print(label47)
    #print(label62)

    print("choose mode [1: label47 | 2: label62 ]")
    mode = input()

    if mode =="1":
        for i in label47:
            belnding(i)

    elif mode=="2":
        for i in label62:
            belnding(i)
    else:
        print("Wrong input")



if __name__== "__main__":
    main()
