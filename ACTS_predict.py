import numpy as np
import scipy.misc
import random
import os
import cv2
import numpngw
import keras.callbacks

from random import shuffle
from tqdm import tqdm
from keras.models import Model,load_model
from keras.layers import Conv2D, UpSampling2D, AveragePooling2D, BatchNormalization, LeakyReLU,Add,Activation,core, Input,concatenate
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from ACTS_functions import create_weighted_binary_crossentropy,MCC,crop_stack

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

##----------------------directories----------------------------------##
data_set_folder = 'C:\\Users\\Christian\\Desktop\\Projects\\PublishACTSCodeData\\ARL_DataSet\\'
otsu_images_folder = data_set_folder+'Images\\'
pred_images_folder = data_set_folder+'Predictions\\'
try:
    os.mkdir(pred_images_folder)
except:
    print('')

training_results_folder = data_set_folder+'Training_results\\'
try:
	os.mkdir(training_results_folder)
except:
	print('')

##-----------------------network laoding-------------------------------##
img_width, img_height, img_channels = 256,256, 1
batch_size = 25

weight_loss = create_weighted_binary_crossentropy()

network = keras.models.load_model(training_results_folder+'Initialized.h5', custom_objects={'weighted_binary_crossentropy': weight_loss,'MCC':MCC})
network.load_weights(training_results_folder+'Trained_weights.h5')
#network.summary()
    
##-------------------------------functions----------------------------##
def crop_stack(img):
    #size=256
    distance=50
    img_stack = []
    predict_img = np.zeros((img.shape[0]+img_width,img.shape[1]+img_height,1))
    for ix in np.arange(0,img.shape[0],distance):
        for iy in np.arange(0,img.shape[1],distance):
            i_img = img[ix:ix+img_width,iy:iy+img_height,:]
            if i_img.shape[0]<img_width or i_img.shape[1]<img_height:
                change_img = np.zeros((img_width,img_height,1))
                change_img[0:i_img.shape[0],0:i_img.shape[1],:]=i_img
                i_img = change_img
             
            predict_img[ix:ix+img_width,iy:iy+img_height,0] = (network.predict_on_batch(np.expand_dims(i_img,axis=0))[0,:,:,0])

    return predict_img

##--------------------------------predictions and saves--------------------------##
specimens = [name for name in os.listdir(otsu_images_folder) if os.path.isdir(otsu_images_folder)]

for ispec in specimens:
    print('Working on '+ispec)
    try:
        os.mkdir(pred_images_folder+ispec)
    except:
        print('')
    
    image_list = [x for x in os.listdir(otsu_images_folder+ispec) if x.endswith('.png')]
    
    for i in tqdm(range(len(image_list))):    
        i_image = np.array(cv2.imread(otsu_images_folder+ispec+'\\'+image_list[i],-1),dtype='float32') 
        pred_image = crop_stack(np.expand_dims(i_image/2**16,axis=-1))
        pred_image = pred_image[:i_image.shape[0],:i_image.shape[1],:]
        numpngw.write_png(pred_images_folder+ispec+'\\'+image_list[i],np.array((pred_image)*255,dtype='uint8'))

