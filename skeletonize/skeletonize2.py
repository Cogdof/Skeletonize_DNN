from skimage import img_as_bool, io, color, morphology, img_as_uint
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.morphology import medial_axis, skeletonize, thin
import os, cv2
import skimage

#원본데이터 경로
path = "/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/dataset/data/Train"
#결과데이터 경로
result_path = "/home/mll/v_mll3/OCR_data/dataset/single_character_dataset/dataset/after_skeletonize/Train"
#하위리스트





files = []
exts = ['jpg', 'png', 'jpeg', 'JPG']
for parent, dirnames, filenames in os.walk(path):
    for filename in filenames:
        for ext in exts:
            if filename.endswith(ext):
                files.append(os.path.join(parent, filename))
                break

print('Find {} images'.format(len(files)))
print('working on {} folder '.format(path))
file_list = os.listdir(path)


for i in range(0,len(files)):
    img = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)       #original
    #img = cv2.imread(files[i])
    thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
    blur = cv2.GaussianBlur(thr, (3, 3), 0)
    image = invert(img_as_bool(blur))
    out = skeletonize(image)

    #skeleton_lee = skeletonize(image, method='lee')

    #skel, distance = medial_axis(image, return_distance=True)

    out = thin(image, max_iter=1)
    #out3 = distance * skel
    #out4 = skeleton_lee

    #skeletonzie한 결과 저장
    files_dir = files[i].split('/')
    output_dir = result_path + files_dir[8]
    if not (os.path.isdir(result_path)):  # 새  파일들을 저장할 디렉토리를 생성
        os.makedirs(os.path.join(result_path))
    io.imsave(output_dir, img_as_uint(out))
    '''
    f, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5)
    ax0.imshow(img, cmap='gray')
    ax0.axis('off')
    ax0.set_title('original', fontsize=20)

    ax1.imshow(out, cmap='gray')
    ax1.axis('off')
    ax1.set_title('skeleton', fontsize=20)

    ax2.imshow(out2, cmap='gray')
    ax2.axis('off')
    ax2.set_title('thin', fontsize=20)

    ax3.imshow(out3, cmap='magma')
    #ax3.imshow(out3, cmap='gray')
    ax3.contour(image, [0.5], color='w')
    ax3.axis('off')
    ax3.set_title('medial_axis', fontsize=20)

    ax4.imshow(out4, cmap='gray')
    ax4.axis('off')
    ax4.set_title('skeleton_lee', fontsize=20)
    #출력 결과보려면 plt.show() 주석처리 제거
    #plt.show()

'''

