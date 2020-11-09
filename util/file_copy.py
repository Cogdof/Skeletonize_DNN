import os
import shutil


'''
# copy images ans seperate each folder for train/test

'''


befor_dir = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/dataset/skeletonized_character_Dataset_1021/Train'
after_dir = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/dataset/skeletonized_character_Dataset_1021/Validation'

befor_file_list = os.listdir(befor_dir)

if not(os.path.isdir(after_dir)):                 #새  파일들을 저장할 디렉토리를 생성
    os.makedirs(os.path.join(after_dir))


for label_folder in befor_file_list:
      # 10%

    count = 0;

    if not (os.path.isdir(after_dir+'/'+label_folder)):  # 새  파일들을 저장할 디렉토리를 생성
        os.makedirs(os.path.join(after_dir+'/'+label_folder))

    image_list = os.listdir(befor_dir+'/'+label_folder)

    test_image_num = len(image_list) * 0.4      # each label images's 5%

    for image in image_list:

        if count < test_image_num:
            shutil.move(befor_dir+'/'+label_folder+'/'+image , after_dir+'/'+label_folder+'/'+image)
            count = count + 1
            print(
                befor_dir + '/' + label_folder + '/' + image + ' \n are copy to \n' + after_dir + '/' + label_folder + '/' + image)
        else :
            break;
