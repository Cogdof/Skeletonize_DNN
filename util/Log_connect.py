word_label_log = open("/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeB_label.txt",'r')
word_predict_deep_log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeB/total_Teb.txt','r')

save_log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc_clear_Deep/DT_detect_not_deep.txt', 'r')


word_label = {}
word_predict_label = {}
log_list =[]
#word label add dict
for line in word_label_log:
    if "/home" in line:
        label = line.split("\t")[1].strip()
        file1 = line.split("\t")[0].split("/")[9]
        file2 = line.split("\t")[0].split("/")[10]
        file = file1+"_"+file2

        word_label.update({file:label})

# DT_log cleansing
for line in word_predict_deep_log:
    if "/home" in line:
        label = line.split("\t")[1].strip()
        file1 = line.split("\t")[0].split("/")[8]
        file2 = line.split("\t")[0].split("/")[9]
        file = file1+"_"+file2

        word_predict_label.update({file:label})
# DT_log cleansing


for line in save_log:
    if "True" in line:

        label = line.split("\t")[1].strip()
        file = line.split(":")[0].strip()
        list = [file, label]
        log_list.append(list)

for i in save_log:
    print(i)