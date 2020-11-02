import numpy as np
import cv2



foreground = cv2.imread('/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/font1.png',cv2.IMREAD_UNCHANGED)
background = cv2.imread('/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/generated_char_img/1.jpeg')  # -1 loads with transparency




added_image = cv2.addWeighted(background,0.4,foreground,0.1,0)

cv2.imshow("Composited image", added_image)
cv2.waitKey(0)