

log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc/label.txt', 'r')

sort_log = open('/home/mll/v_mll3/OCR_data/final_dataset/dataset/TeBc/label_s.txt', 'w')


line_list = []



for line in log:
    line_list.append(line)

log.close()
line_list.sort()

for line in line_list:
    sort_log.write(line)


sort_log.close()