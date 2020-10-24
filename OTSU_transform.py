import os
import cv2
from skimage.filters import threshold_otsu
import numpy as np
import numpngw
from tqdm import tqdm
image_folder = 'C:\\Users\\Christian\\Desktop\\Projects\\PublishACTSCodeData\\RawData\\C4\\Images\\'

image_list = [x for x in os.listdir(image_folder) if x.endswith('.tif')]
# compute mean across data stack
mean_list = []
std_list = []
print('Getting Stack Statistics')
for i_name in image_list[:]:
    i_image = np.array(cv2.imread(image_folder+i_name,-1),dtype='float32') 
    thresh = threshold_otsu(i_image)
    mean_list.append(np.mean(i_image[i_image>thresh]))
    std_list.append(np.std(i_image[i_image>thresh]))
# covert to otsu image
stack_mean = np.mean(np.array(mean_list))
stack_std = np.mean(np.array(std_list))
try:
    os.mkdir(image_folder+'Otsu_stack')
except:
    print('')
print('Saving stack')    
for i in tqdm(range(len(image_list))):    
    i_image = np.array(cv2.imread(image_folder+image_list[i],-1),dtype='float32')     
    new_image = (i_image-stack_mean)/stack_std
    numpngw.write_png(image_folder+'Otsu_stack'+'\\'+image_list[i][:-3]+'png',np.array((new_image+30)*2**16/35,dtype='uint16'))
    
