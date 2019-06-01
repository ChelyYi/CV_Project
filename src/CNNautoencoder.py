
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization,Activation
from keras.models import Model
import numpy as np

x_train = np.load("../data/x_train.npy")
y_train= x_train.copy()

def upsample_conv(input,filter,model_name,kernel_size=(3,3),sample_size=(2,2),padding="same",activation="relu"):

    upsample = UpSampling2D(sample_size,name=model_name+'_Upsample')(input)
    conv_out = Conv2D(filter,kernel_size,name=model_name+"_Conv",padding=padding,activation=activation)(upsample)
    return conv_out


def conv_maxpooling(input,filter,model_name,kernel_size=(3,3),pool_size=(2,2),padding="same",activation="relu"):
    conv_out = Conv2D(filter,kernel_size,name=model_name+"_Conv",padding=padding,activation=activation)(input)
    pool_out = MaxPooling2D(pool_size,name=model_name+"_Maxpooling")(conv_out)
    return pool_out


def encoder(num_of_layers,filter_num_list,input):

    enc_in = input
    for i in range(num_of_layers):
        enc_in = conv_maxpooling(enc_in,filter_num_list[i],model_name="Encoder"+str(i))
    return enc_in


def decoder(num_of_layers,filter_num_list,input):

    dec_in = input
    for i in range(num_of_layers):
        dec_in = upsample_conv(dec_in,filter_num_list[i],model_name="Decoder"+str(i))
    return dec_in


def autoencoder(num_of_layers,filter_num_ist):

    input = Input(shape=(256,256,3))
    enc_out = encoder(num_of_layers,filter_num_ist,input)
    dec_out = decoder(num_of_layers,filter_num_ist,enc_out)

    model = Model(input,dec_out)
    model.compile(loss="mean_absolute_error",optimizer="adam")
    return model

if __name__ == "main":
    model = autoencoder(3,[16,8,3])
    model.summary()
    model.fit(x=x_train,y=y_train,batch_size=32,epochs=30,validation_split=0)
    model.save("CNNautoencoder.h5")
    print("The model has been saved successfully")
