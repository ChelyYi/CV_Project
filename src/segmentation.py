import keras
import tensorflow as tf
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import UpSampling2D,Conv2D,Lambda
from keras.models import Model
import keras.backend as K

def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

x_train = np.load('../data/seg/x_seg_train.npy')
y_train = np.load("../data/seg/y_seg_train.npy")

num_of_class = 21
image_size = 256

def upsample_conv(input,filter,model_name,kernel_size=(3,3),sample_size=(2,2),padding="same",activation="relu"):
    upsample = UpSampling2D(sample_size,name=model_name+'_Upsample')(input)
    conv_out = Conv2D(filter,kernel_size,name=model_name+"_Conv",padding=padding,activation=activation)(upsample)
    return conv_out


vgg = VGG16(weights='imagenet',include_top = False,input_shape=(image_size,image_size,3))

#image_feature = vgg.output
#result = Conv2D(kernel_size=(3,3),filters=num_of_class,name="32Deconv",padding='same')(UpSampling2D(size=(32,32),name='32Upsampling')(image_feature))

stage_1 = vgg.get_layer(name="block1_pool").output
stage_2 = vgg.get_layer(name="block2_pool").output
stage_3 = vgg.get_layer(name="block3_pool").output
stage_4 = vgg.get_layer(name="block4_pool").output
stage_5 = vgg.get_layer(name="block5_pool").output

vgg.summary()
upsample_stage_5 = upsample_conv(stage_5,filter=512,model_name="Upsample_stage_5")
merge_1 = Lambda(lambda x: x[0]+x[1])([upsample_stage_5,stage_4])
# merge_1  = upsample_stage_5 + stage_4
upsample_merge_1 = upsample_conv(merge_1,filter=256,model_name="Upsample_merge_1")
merge_2 = Lambda(lambda x: x[0] + x[1])([upsample_merge_1,stage_3])
# merge_2 = upsample_merge_1 + stage_3
output = upsample_conv(merge_2,num_of_class,sample_size=(8,8),model_name="Upsample_merge_2")

model = Model(vgg.input,output)

def cross_entropy_loss(y_pred,y_true):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_pred,logits=y_true)

model.summary()
#output_target = tf.placeholder(dtype="int32",shape=(None,256,256))
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss=softmax_sparse_crossentropy_ignoring_last_label)#target_tensors=[output_target])

model.fit(x=x_train,y=y_train,batch_size=32,epochs=30,validation_split=0.1)

model.save('..model/segmentation1.h5')