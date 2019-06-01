import keras
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import UpSampling2D,Conv2D
from keras.models import Model


x_train = np.load('../data/seg/x_seg_train.npy')
y_train = np.load("../data/seg/y_seg_train.npy")
image_size = 256

vgg = VGG16(weights='imagenet',include_top = False,input_shape=(image_size,image_size,3))
vgg.summary()
image_feature = vgg.output
result = Conv2D(kernel_size=(3,3),filters=3,name="32Deconv",padding='same')(UpSampling2D(size=(32,32),name='32Upsampling')(image_feature))

model = Model(vgg.input,result)
for layer in model.layers[:18]:
    layer.trainable = False

model.summary()
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy')
model.fit(x=x_train,y=y_train,batch_size=10,epochs=30,validation_split=0.1)

model.save('..model/segmentation1.h5')