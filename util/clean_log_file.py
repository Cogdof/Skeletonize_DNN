import os

#log_path = '/home/mll/v_mll3/OCR_data/final_dataset/dataset/Test/IC13/log/log_demo_result.txt'

total_deep_result_log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc/TeBc_gt/log_demo_result.txt', 'r')
total_gt_log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc/TeBc_gt/total_gt.txt', 'r')
save_log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc/TeBc_gt/save_log2.txt', 'w')

unsigned_count =0
multi_count =0
equal_count = 0 # deep  이알고있 음
total ={}

for line in total_gt_log:
    file = line.split("\t")[0]
    label = line.split("\t")[1].strip()
    total.update({file : label})


for line in total_deep_result_log:
    if "/home" in line:
        file = line.split("\t")[0]
        file_dir = file.split("/")[11]+"/"+file.split("/")[12]
        label = line.split("\t")[1]
        label = label.strip()
        if total.get(file_dir) == label or total.get(file_dir) == label.lower():
            equal_count+=1
            print(line)
        else :
            save_log.write(file_dir +"\t"+label +"\t" +total.get(file_dir) +"\n")
            print(">> "+line)
        if label== "#":
            unsigned_count +=1
        if len(label) >1:
            multi_count +=1



total_deep_result_log.close()
total_gt_log.close()
save_log.close()
print(equal_count)
print(unsigned_count)
print(multi_count)