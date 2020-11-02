import cv2
import os
import PIL.Image as Image
from PIL import ImageFilter
import numpy as np
import random
'''
------------------------------------
# [20.10.23 build]
# Generate synth char data.

merge wild background img and char font img.

img type : RGB, HSB, YUB ?

label :
ver1 : 47 0-9 a-z + @ (ABDEFGHNQRT)
ver2 : 62 0-9 a-z A-Z

[Lastest update] : 2020.11.01

------------------------------------
'''
file_dir  = "/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/v1/"
font_dir47 = "/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/Fnt_transparent_47/"
font_dir62 = "/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/Fnt_transparent_62/"
bg_dir = "/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/Background/"
save_dir = "/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/generate_img/"

def color_convert(img):
    #change font's color randomly

    random1 = random.randint(0, 255)
    random2 = random.randint(0, 255)
    random3 = random.randint(0, 255)

    for i in range(0, img.size[0]):
        for j in range(0, img.size[1]):
            rgb = img.getpixel((i, j))
            if rgb[0] == 0: #and rgb[1]==0 and rgb[3]==0:
                rgb_r = (random1-rgb[0], random2 - rgb[1], random3 - rgb[2])
                img.putpixel((i, j), rgb_r)
    return img



def belnding(label, label_count):
    # merge font and background

    if label_count ==47:
        font_dir = font_dir47
    else:
        font_dir = font_dir62
    font_dir = font_dir + "/" + label
    font_list = os.listdir(font_dir)

    bg_list = os.listdir(bg_dir)



    count=0

    for a in bg_list:
        bg = Image.open("{}".format(bg_dir+"/"+a), 'r')

        for b in font_list:
            count+=1
            font = Image.open("{}".format(font_dir+"/"+b), 'r')
            text_img = Image.new('RGBA', (font.width, font.height), (0, 0, 0, 0))


            # random count
            # blur 0~5
            # tranperncy : 50 ~ 90%



            # Blur
            blur_count = random.randint(0,3)
            gaussianBlur = ImageFilter.GaussianBlur(blur_count)
            font = font.filter(gaussianBlur)

            # Opacity
            TRANSPARENCY = random.randint(55,95)

            # Color convertinput()
            font = color_convert(font)

            font_mask = font.split()[3].point(lambda i: i * TRANSPARENCY / 100.)

            # font2 = font2.rotate(45)
            # font2 = font2.rotate(90)

            text_img.paste(bg, (0, 0))
            text_img.paste(font, (0, 0), mask=font_mask)
            #text_img.show()

            if not (os.path.isdir(save_dir+label)):  # 새  파일들을 저장할 디렉토리를 생성
                os.makedirs(os.path.join(save_dir+label))

            text_img.save(save_dir+label+"/"+str(label)+"_"+str(count)+".png")
            #print(save_dir+label+"/"+label+"_"+count+".png")
            text_img.close()

    print(label +" is done..")





def main():

    '''
    font = Image.open('/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/font1.png', 'r')
    bg = Image.open('/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/2.jpeg', 'r')
    text_img = Image.new('RGBA', (font.width, font.height), (0, 0, 0, 0))

    # Blur
    blur_count = 3
    gaussianBlur = ImageFilter.GaussianBlur(blur_count)
    font = font.filter(gaussianBlur)

    # Opacity
    TRANSPARENCY = 55

    # Color convert
    font = color_convert(font)

    font_mask = font.split()[3].point(lambda i: i * TRANSPARENCY / 100.)

    # font2 = font2.rotate(45)
    # font2 = font2.rotate(90)

    text_img.paste(bg, (0, 0))
    text_img.paste(font, (0, 0), mask=font_mask)
    text_img.show()
    text_img.save("ball.png", format="png")

    '''





    label47 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R', 'T', 'a', 'b',
               'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
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
        label = 47
        for i in label47:

            belnding(i,label)
            print("working on "+i)
    elif mode=="2":
        label = 62
        for i in label62:
            print("working on " + i)
            belnding(i,label)
    else:
        print("Wrong input")

    print("done!")


if __name__== "__main__":
    main()
