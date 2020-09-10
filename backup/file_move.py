import os
import shutil

befor_dir = '/home/mll/v_mll3/OCR_data/인식_100데이터셋/single_character_Data (사본)/beta_skeletonize/Train'
after_dir = '/home/mll/v_mll3/OCR_data/인식_100데이터셋/single_character_Data (사본)/beta_skeletonize/Test'

befor_file_list = os.listdir(befor_dir)

if not(os.path.isdir(after_dir)):                 #새  파일들을 저장할 디렉토리를 생성
    os.makedirs(os.path.join(after_dir))

test_image_num = 10

for label_folder in befor_file_list:
    count = 0;

    if not (os.path.isdir(after_dir+'/'+label_folder)):  # 새  파일들을 저장할 디렉토리를 생성
        os.makedirs(os.path.join(after_dir+'/'+label_folder))

    image_list = os.listdir(befor_dir+'/'+label_folder)
    for image in image_list:

        if count < test_image_num:
            shutil.copy(befor_dir+'/'+label_folder+'/'+image , after_dir+'/'+label_folder+'/'+image)
            count = count + 1
            print(
                befor_dir + '/' + label_folder + '/' + image + ' \n are copy to \n' + after_dir + '/' + label_folder + '/' + image)
        else :
            break;
