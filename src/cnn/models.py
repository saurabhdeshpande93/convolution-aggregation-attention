
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, concatenate, Lambda, ReLU, MaxPooling2D, Conv2D, ZeroPadding2D,UpSampling2D


class Models():

  def __init__(self):
      pass

  def CNN(self,input_shape):
      inputs = Input(shape=input_shape)
      ##
      conv1 = Conv2D(64, (3, 3), padding='same',data_format='channels_first')(inputs)
      conv1 = ReLU()(conv1)

      conv1 = Conv2D(64, (3, 3), padding='same',data_format='channels_first')(conv1)
      conv1 = ReLU()(conv1)
      pool1 = MaxPooling2D((2,2),data_format='channels_first')(conv1)
      ##
      conv2 = Conv2D(128, (3,3), padding='same',data_format='channels_first')(pool1)
      conv2 = ReLU()(conv2)

      conv2 = Conv2D(128, (3,3), padding='same',data_format='channels_first')(conv2)
      conv2 = ReLU()(conv2)
      pool2 = MaxPooling2D((2,2),data_format='channels_first')(conv2)

      ##
      conv3 = Conv2D(256, (3,3), padding='same',data_format='channels_first')(pool2)
      conv3 = ReLU()(conv3)

      conv3 = Conv2D(256, (3,3), padding='same',data_format='channels_first')(conv3)
      conv3 = ReLU()(conv3)
      pool3 = MaxPooling2D((2,2),data_format='channels_first')(conv3)
      ##

      conv4 = Conv2D(256, (3,3), padding='same',data_format='channels_first')(pool3)
      conv4 = ReLU()(conv4)

      conv4 = Conv2D(256, (3,3), padding='same',data_format='channels_first')(conv4)
      conv4 = ReLU()(conv4)

      ##
      up1 = UpSampling2D(size=(2,2),data_format='channels_first')(conv4)
      up1 = concatenate([conv3,up1],axis=1)
      conv5 = Conv2D(256, (3,3), padding='same',data_format='channels_first')(up1)
      conv5 = ReLU()(conv5)

      conv5 = Conv2D(256, (3, 3),padding='same',data_format='channels_first')(conv5)
      conv5 = ReLU()(conv5)
      #
      up2 = UpSampling2D(size=(2,2),data_format='channels_first')(conv5)
      up2 = concatenate([conv2,up2],axis=1)
      conv6 = Conv2D(128, (3,3), padding='same',data_format='channels_first')(up2)
      conv6 = ReLU()(conv6)

      conv6 = Conv2D(128, (3,3), padding='same',data_format='channels_first')(conv6)
      conv6 = ReLU()(conv6)
      #
      up3 = UpSampling2D(size=(2,2),data_format='channels_first')(conv6)
      up3 = concatenate([conv1,up3],axis=1)
      conv7 = Conv2D(64, (3,3), padding='same',data_format='channels_first')(up3)
      conv7 = ReLU()(conv7)

      conv7 = Conv2D(64, (3,3), padding='same',data_format='channels_first')(conv7)
      conv7 = ReLU()(conv7)
      #

      conv8 = Conv2D(2, (1,1),  activation= None, padding= 'same',data_format='channels_first')(conv7)

      UNET = Model(inputs=inputs, outputs=conv8)

      return UNET


  def MAgNET(self):
      pass #To be added after the acceptance of MAgNET paper.
