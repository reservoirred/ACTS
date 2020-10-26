import os
import cv2
import numpngw
import numpy as np

from skimage.filters import threshold_otsu
from tqdm import tqdm
##-----------------------------------Location of complete data repository------------------------------------####
data_set_folder = 'C:\\Users\\Christian\\Desktop\\Projects\\PublishACTSCodeData\\ARL_DataSet\\'

##--------------------------Directories------------------------------------##

raw_images_folder = data_set_folder+'Raw\\'
otsu_images_folder = data_set_folder+'Images\\'

try:
	os.mkdir(otsu_images_folder)
except:
	print('')

##---------------------------OTSU transformation-------------------------------##

specimens = [name for name in os.listdir(raw_images_folder) if os.path.isdir(raw_images_folder) ]

for ispec in specimens:
    print('Working on '+ispec)
    image_list = [x for x in os.listdir(raw_images_folder+ispec) if x.endswith('.tif')]
    # compute mean across data stack
    mean_list = []
    std_list = []
    for i_name in image_list[:]:
        i_image = np.array(cv2.imread(raw_images_folder+ispec+'\\'+i_name,-1),dtype='float32') 
        thresh = threshold_otsu(i_image)
        mean_list.append(np.mean(i_image[i_image>thresh]))
        std_list.append(np.std(i_image[i_image>thresh]))
    # covert to otsu image
    stack_mean = np.mean(np.array(mean_list))
    stack_std = np.mean(np.array(std_list))
    try:
        os.mkdir(otsu_images_folder+ispec)
    except:
        print('')
    
    for i in tqdm(range(len(image_list[:]))):    
        i_image = np.array(cv2.imread(raw_images_folder+ispec+'\\'+image_list[i],-1),dtype='float32')     
        new_image = (i_image-stack_mean)/stack_std
        ind_file = image_list[i].find('.',-5, -1)
        numpngw.write_png(otsu_images_folder+ispec+'\\'+image_list[i][:ind_file]+'.png',np.array((new_image+30)*2**16/35,dtype='uint16'))

