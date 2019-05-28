# import tensorflow.keras as keras
#import keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,\
                                    Flatten,Reshape,UpSampling2D,Input,\
                                    BatchNormalization,Activation
from tensorflow.keras.models import Model
import numpy as np

x_train = np.load("../data/x_train.npy")
y_ = x_train.copy()

def conv(input,filter_num,strides=1,kernel_size=(3,3),pool_size=(2,2)):
    conv1 = Conv2D(filter_num,kernel_size,strides)(input)
    pooling_out = MaxPooling2D(pool_size=pool_size)(conv1)
    output = Activation("relu")(pooling_out)
    return output

def trans_conv(input,filter_num,strides=1,kernel_size=(3,3),sampling_size=(2,2)):
    sampling_out = UpSampling2D(sampling_size)(input)
    conv_out = Conv2D(filter_num,kernel_size,strides,activation="relu")(sampling_out)
    return conv_out


input = Input(shape=(214,214,3))
batch_norm = BatchNormalization()(input)

conv1_out = conv(batch_norm,32)
conv2_out = conv(conv1_out,16)
conv3_out = conv(conv2_out,8)
conv4_Out = conv(conv3_out,4,strides=2)

flatten = Flatten()(conv4_Out)
compression = Dropout(0.2)(Dense(100)(flatten))
dense_2 = Reshape((16,16,1))(Dense(256)(compression))

dec_1 = trans_conv(dense_2,filter_num=2,kernel_size=(2,2))
dec_2 = trans_conv(dec_1,filter_num=3,kernel_size=(2,2))
dec_3 = trans_conv(dec_2,filter_num=3,kernel_size=(5,5))
dec_4 = trans_conv(dec_3,filter_num=3,kernel_size=(3,3))
output = Conv2D(3,(3,3),strides=1,activation="relu")(dec_4)

model = Model(input,output)
model.compile(loss="mean_squared_error",
              optimizer ="adam",
              metrics=['accuracy'])
model.summary()

model.fit(x=x_train,y=y_,batch_size=32,epochs=10,validation_split=0.2)
model.save("../model/good_model.h5")


#model.add(Conv2D(32,(3,3),activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(16,(3,3),strides=1,activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(8,(3,3),strides=1,activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))


#model.add(Conv2D(4,(3,3),strides=2,activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Flatten())
#model.add(Dense(100))
#model.add(Dropout(0.2))

#model.add(Dense(256))
#model.add(Reshape((16,16,1)))

#model.add(UpSampling2D(size=(2,2)))
#model.add(Conv2D(2,(3,3),strides=1,activation="relu"))

#model.add(UpSampling2D(size=(2,2)))
#model.add(Conv2D(3,(2,2),strides=1,activation="relu"))

#model.add(Conv2D(3,(5,5),strides=1,activation="relu"))
#model.add(UpSampling2D(size=(2,2)))


#model.add(Conv2D(3,(3,3),strides=1,activation="relu"))
#model.add(UpSampling2D(size=(2,2)))

#model.add(Conv2D(3,(3,3),strides=1,activation="relu"))

#print(model.summary())
#print(x_train.shape)
#model.compile(loss="mean_squared_error",
#              optimizer ="adam",
#             metrics=['accuracy'])

#model.fit(x=x_train,y=y_,batch_size=32,epochs=10,validation_split=0.2)
#model.save("good_model")






