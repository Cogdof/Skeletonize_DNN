import os
import shutil

# Read log file, move each character label folder



label47 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R', 'T', 'a', 'b',
                 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc_clear_Deep/Test/'
image_dir = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc_notDeep/'
log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc_clear_Deep/save_log.txt', 'r')

for i in label47:
    folder_dir = dir + i
    if not (os.path.isdir(dir + i)):
        os.makedirs(os.path.join(dir + i))

count =0
for line in log:

    img_dir = line.split("\t")[0]
    filename = img_dir.replace("/","_")
    label = line.split("\t")[2].strip()
    label = label.strip()
    #score = float(line.split("\t")[2].strip())

    if len(label)==1 :#and score > 0.8:
        if label in label47 or label.lower() in label47:
            folder_dir = dir+label
            if not (os.path.isdir(folder_dir)):
                folder_dir = dir+label.lower()
            shutil.copy(image_dir+filename, folder_dir + '/'+filename)
            count +=1
        else :
            print(filename)


print(count)
log.close()
print("done! ")