import os
import shutil

# Read log file, move each character label folder


# label_47
#label47 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R', 'T', 'a', 'b',
#               'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

label_dir = "/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrGc_label"
dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrGc_clear/'

label47 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R', 'T', 'a', 'b',
                 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


folderlist = os.listdir(label_dir)
for fold in folderlist:

    log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TrGc_label/{}/log_high.txt'.format(fold), 'r')

    print(fold +" is working")

    for i in label47:
        folder_dir = dir + i
        if not (os.path.isdir(dir + i)):
            os.makedirs(os.path.join(dir + i))


    for line in log:
        if "/home" in line:
            img_dir = line.split("\t")[0]
            label = line.split("\t")[1]
            label = label.strip()
            score = float(line.split("\t")[2].strip())
            if len(label)==1 and score > 0.8:
                if label in label47 or label.lower() in label47:
                    folder_dir = dir+label
                    if not (os.path.isdir(folder_dir)):
                        folder_dir = dir+label.lower()
                    shutil.copy(img_dir, folder_dir + '/')




    log.close()
print("done! ")