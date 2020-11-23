import os



gt_file = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/Test/IC13/log/gt.txt','r')
log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/Test/IC13/log/log_cleansing.txt', 'r')
log_gt = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/Test/IC13/log/log_gt.txt', 'w')

log_lines = log.readlines()
gt_lines = gt_file.readlines()
count = 0
for i in range(0, len(log_lines )):
    predict_label = log_lines[i].split("\t")[1]
    label = gt_lines[i].split("\t")[1]
    label = label.strip()
    predict_label = predict_label.strip()

    if label != predict_label:
        count+=1
        #print(log_lines[i].split("\t")[0] + "\t" + log_lines[i].split("\t")[1] + "\t" + label)
        log_gt.write(log_lines[i].split("\t")[0] + "\t" + log_lines[i].split("\t")[1] + "\t" + label + "\n")

log_gt.write("\n"+ str(count) + "\n")
log_gt.close()