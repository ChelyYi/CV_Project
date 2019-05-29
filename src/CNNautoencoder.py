# import tensorflow.keras as keras
import keras
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,\
                                    Flatten,Reshape,UpSampling2D,Input,\
                                    BatchNormalization,Activation
from keras.models import Model
import numpy as np

x_train = np.load("../data/x_train.npy")
y_train= x_train.copy()

def upsample_conv(input,filter,kernel_size=(3,3),sample_size=(2,2),padding="same",activation="relu"):
    upsample = UpSampling2D(sample_size)(input)
    conv_out = Conv2D(filter,kernel_size,padding=padding,activation=activation)(upsample)
    return conv_out

def conv_maxpooling(input,filter,kernel_size=(3,3),pool_size=(2,2),padding="same",activation="relu"):
    conv_out = Conv2D(filter,kernel_size,padding=padding,activation=activation)(input)
    pool_out = MaxPooling2D(pool_size)(conv_out)
    return pool_out

input = Input(shape=(256,256,3))

encoder_layer_1 = conv_maxpooling(input,16)
enc_out = conv_maxpooling(encoder_layer_1,3)

decoder_layer_1 = upsample_conv(enc_out,16)
dec_out = upsample_conv(decoder_layer_1,3)


autoencoder = Model(input,dec_out)
autoencoder.compile(loss = "mean_absolute_error",
              optimizer="adam")
autoencoder.summary()
autoencoder.fit(x=x_train,y=y_train,batch_size=32,epochs=30,
          validation_split=0)
autoencoder.save("good_model_version_3_depth_3.h5")
print("model has been saved successfully")