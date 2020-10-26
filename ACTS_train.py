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
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

## ------------------------Inputs-----------------------------------##
data_set_folder = 'C:\\Users\\Christian\\Desktop\\Projects\\PublishACTSCodeData\\ARL_DataSet\\'
## inputs ##
img_width, img_height, img_channels = 256,256, 1
batch_size = 25
epochs = 200
RegTerm = 0.001
valid_per = .2
##--------------------------Directories------------------------------------##
images_folder = data_set_folder+'Images\\'
labels_folder = data_set_folder+'Labels\\'
training_results_folder = data_set_folder+'Training_results\\'
try:
	os.mkdir(training_results_folder)
except:
	print('')

##-------------------------function list------------------------------------##

def get_sample(sample_name,aug):
    img = np.expand_dims(np.array(cv2.imread(images_folder+sample_name,-1),dtype='float32')/2**16,axis=-1);
    lbl = np.array(cv2.imread(labels_folder+sample_name),dtype='float32')/255;
    lbl[:,:,1] = 1-lbl[:,:,1]
    
    kx = random.sample(range(5,img.shape[0]-5-img_width,1),1)[0]
    ky = random.sample(range(5,img.shape[1]-5-img_height,1),1)[0]

    img = img[kx:kx+img_width,ky:ky+img_height,:]
    lbl = lbl[kx:kx+img_width,ky:ky+img_height,:]
    
    if aug:
        k = random.sample(range(0,360,90),1)[0]
        M = cv2.getRotationMatrix2D((img_width/2,img_height/2), k, 1)
        img = cv2.warpAffine(img, M, (img_width,img_height)) 
        lbl = cv2.warpAffine(lbl, M, (img_width,img_height),cv2.INTER_NEAREST,cv2.BORDER_CONSTANT,borderValue=(0, 0, 0))
        if (random.sample(range(0,2),1)[0] % 2) == 0:
            img = np.fliplr(img)
            lbl = np.fliplr(lbl)
        elif (random.sample(range(0,2),1)[0] % 2) == 0:
            img = np.flipud(img)
            lbl = np.flipud(lbl)
        img = np.expand_dims(img,axis=-1)
        
    img = np.clip(img,0,1) 
    
    #get boundary mask
    sobelx = cv2.Sobel(lbl[:,:,0],cv2.CV_32F,1,0,ksize=3)
    sobely = cv2.Sobel(lbl[:,:,0],cv2.CV_32F,0,1,ksize=3)
    boundary_location = np.sqrt(sobelx**2+sobely**2)
    
    kernel = np.ones((5,5),np.float32)/9
    dst = cv2.filter2D(np.array(boundary_location),-1,kernel)+1
    dst[:,-3:]=1
    dst[:,:3]=1
    dst[:3,:]=1
    dst[-3:,:]=1
    
    lbl[:,:,-1] = dst
    
    return img,lbl
    
def batch_generator(sample_list,batch_size,aug):
    while True:
        batch_filenames = np.random.choice(sample_list,batch_size)
        batch_input = []
        batch_output = []
        for i_sample in batch_filenames:
            i_array,i_class = get_sample(i_sample,aug)
            batch_input+=[i_array]
            batch_output+=[i_class]
        batch_input = np.array(batch_input,dtype='float32')
        batch_output = np.array(batch_output,dtype='float32')
        yield (batch_input,batch_output)
        
#imgb,lblb = next(batch_generator(sample_list,batch_size,True))
#fig, axs = plt.subplots(1, 3, figsize=(8, 4), sharey=True)
#axs[0].imshow(imgb[0,:,:,0],vmin=0,vmax=1)
#axs[1].imshow(lblb[0,:,:,0])  
#axs[2].imshow(lblb[0,:,:,2]) 

def MCC(y_true_n,y_pred_n):
    y_true = K.cast(K.expand_dims(K.argmax(y_true_n[:,:,:,0:2],axis=-1),axis=-1),dtype='float32')
    y_pred = K.cast(K.expand_dims(K.argmax(y_pred_n,axis=-1),axis=-1),dtype='float32')
    tp = K.sum(K.round(K.clip(y_true,0,1))*K.clip(y_pred,0,1))
    tn = K.sum(K.round(1-K.clip(y_true,0,1))*(1-K.clip(y_pred,0,1)))
    fp = K.sum(K.round(1-K.clip(y_true,0,1))*(K.clip(y_pred,0,1)))
    fn = K.sum(K.round(K.clip(y_true,0,1))*(1-K.clip(y_pred,0,1)))
    MCC = ((tp*tn)-(fp*fn))/K.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return MCC

def create_weighted_binary_crossentropy():
    def weighted_binary_crossentropy(y_true, y_pred):
        b_ce = K.binary_crossentropy(y_true[:,:,:,0:2],y_pred[:,:,:,0:2])
        # Apply the weights
        class0weight = 1
        class1weight = 30
        mask0 = K.ones_like(y_true[:,:,:,0])*y_true[:,:,:,0]*class0weight
        mask1 = K.ones_like(y_true[:,:,:,1])*y_true[:,:,:,1]*class1weight
        weight_vector = mask0+mask1
        weighted_b_ce = b_ce*weight_vector*y_true[:,:,:,-1]
        return K.mean(b_ce)
    return weighted_binary_crossentropy

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.metric = []
    #def on_train_end(self, logs={}):
    #def on_epoch_begin(self, logs={}):    
    #def on_epoch_end(self, logs={}):   
    #def on_batch_begin(self, logs={}):
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.metric.append(logs.get('MCC'))  

def Classifylayer(input_layer,numel_classes):
    CL_1 = Conv2D(numel_classes,(1,1),padding='same',activation=None,kernel_regularizer=regularizers.l2(RegTerm),bias_regularizer=regularizers.l2(RegTerm))(input_layer)
    CL_2 = LeakyReLU(alpha=.3)(CL_1)
    CL_3 = core.Activation('softmax')(CL_2)
    return CL_3

def convf(input_layer,numel_filters):
    CL_1 = Conv2D(numel_filters, (3,3), padding='same',activation=None,kernel_regularizer=regularizers.l2(RegTerm),bias_regularizer=regularizers.l2(RegTerm))(input_layer)
    CL_2 = BatchNormalization(axis=-1,momentum = .5)(CL_1)
    CL_3 = LeakyReLU(alpha=.3)(CL_2)
    CL_4 = Conv2D(numel_filters, (3,3), padding='same',activation=None,kernel_regularizer=regularizers.l2(RegTerm),bias_regularizer=regularizers.l2(RegTerm))(CL_3)
    CL_5 = BatchNormalization(axis=-1,momentum = .5)(CL_4)
    CL_6 = Add()([CL_1,CL_5])
    CL_7 = LeakyReLU(alpha=.3)(CL_6)
    return CL_7

def downsample(input_layer):
    CL_1 = AveragePooling2D(pool_size=(2,2),padding='valid')(input_layer)
    return CL_1

def upsample(input_layer):
    CL_1 = UpSampling2D(size=(2,2))(input_layer)
    return CL_1

## ---------------------------------------- Build Network----------------------------------------##
Image_input = Input(shape = (img_width, img_height,  img_channels)) 

Encode_Stage_1_1 = convf(Image_input,6)
Encode_Stage_1 = convf(Encode_Stage_1_1,6) 
Encode_Stage_1_downsample = downsample(Encode_Stage_1) 
Encode_Stage_2 = convf(Encode_Stage_1_downsample,12) 
Encode_Stage_2_downsample = downsample(Encode_Stage_2) 
Encode_Stage_3 = convf(Encode_Stage_2_downsample,24) 
Encode_Stage_3_downsample = downsample(Encode_Stage_3) 

Central_stage = convf(Encode_Stage_3_downsample,48)

Decode_Stage_3_upsample = upsample(Central_stage)
Decode_Stage_3 = convf(concatenate([Decode_Stage_3_upsample,Encode_Stage_3]),24)
Decode_Stage_2_upsample = upsample(Decode_Stage_3) 
Decode_Stage_2 = convf(concatenate([Decode_Stage_2_upsample,Encode_Stage_2]),12) 
Decode_Stage_1_upsample = upsample(Decode_Stage_2) 
Decode_Stage_1 = convf(concatenate([Decode_Stage_1_upsample,Encode_Stage_1]),6) 

Decode_Stage_1_1 = convf(Decode_Stage_1,6) 

Class_Stage = Classifylayer(Decode_Stage_1_1,2) 

network = Model(inputs=[Image_input],outputs=[Class_Stage])

weight_loss = create_weighted_binary_crossentropy()

network.compile(loss=weight_loss,optimizer='adam',metrics=[MCC])
#network.summary()
network.save(training_results_folder+'Initialized.h5')
network.save_weights(training_results_folder+'Initialized_weights.h5')

## --------------------------------- Train Network --------------------------------------##
sample_list = [os.path.join(path, name)[len(images_folder):] for path, subdirs, files in os.walk(images_folder) for name in files]

indexes_valid = random.sample(range(0,len(sample_list)),int(valid_per*float(len(sample_list))))
indexes_train = [x for x in range(0,len(sample_list)) if x not in indexes_valid]
training_list = [x for ind,x in enumerate(sample_list) if ind in indexes_train]
valid_list = [x for ind,x in enumerate(sample_list) if ind in indexes_valid]

num_batch_calls = (int(float(len(training_list)+1)/float(batch_size)))
valid_batch_calls = (int(float(len(valid_list)+1)/float(batch_size))-1)

history_call = LossHistory()

val_stop_call = EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=1,mode='auto', restore_best_weights=True)
reduce_lr_call = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=.000002)

traininghistory = network.fit_generator(batch_generator(sample_list,batch_size,True),
                      steps_per_epoch = num_batch_calls,epochs = epochs,
                     validation_data = batch_generator(sample_list,batch_size,True),
                      validation_steps = valid_batch_calls ,
                      callbacks=[history_call,val_stop_call,reduce_lr_call],verbose=1)

network.save_weights(training_results_folder+'Trained_weights.h5')
np.savetxt(training_results_folder+'TrainLoss.txt',history_call.loss)
np.savetxt(training_results_folder+'TrainMCC.txt',history_call.metric)
np.savetxt(training_results_folder+'ValLoss.txt',traininghistory.history['val_loss'])
np.savetxt(training_results_folder+'ValMCC.txt',traininghistory.history['val_MCC'])

##---------------------------------------Classification report------------------------------##
print('Calculating stats on validation set')
i_pred = []
i_true = []
for i in tqdm(range(len(valid_list))):
    img,lbl = get_sample(valid_list[i],False)
    pred = network.predict_on_batch(np.expand_dims(img,axis=0))

    i_true.extend(np.argmax(lbl[:,:,0:2],axis=-1).flatten())
    i_pred.extend(np.argmax(pred,axis=-1).flatten())

print('Saving report')
report = classification_report(i_true,i_pred,target_names=['Part','Pore'])
print(report)
f = open(training_results_folder+'ClassificationReport.txt','w')
print(report, file=f)
f.close() 
