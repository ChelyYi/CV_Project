import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import UpSampling2D,Conv2D,Dropout,ZeroPadding2D,\
    BatchNormalization,Activation,concatenate,GlobalAveragePooling2D,Dense
from keras.models import Model


image_size = 256
class_num = 21
learning_rate = 0.0001
epochs = 50

def fcn_32(image_feature):
    fc6 = Conv2D(kernel_size=(7,7),filters=4096,name="fullConv1",padding='same',activation='relu')(image_feature)
    fc6 = Dropout(0.5)(fc6)
    fc7 = Conv2D(kernel_size=(1,1),filters=4096,name="fullConv2",padding='same',activation='relu')(fc6)
    fc7 = Dropout(0.5)(fc7)
    score = Conv2D(kernel_size=(1,1),filters=class_num,name='scoreConv',padding='same',kernel_initializer='he_normal')(fc7)
    result = Conv2D(kernel_size=(16,16),filters=class_num,name="32Deconv",padding='same',use_bias=False)\
        (UpSampling2D(size=(32,32),name='32Upsampling',interpolation='bilinear')(score))

    return result

def segnet_decoder_block(input,filter,kernel_size=(3,3),padding='same',upsample=False):
    if upsample: # do upsample first
        input = UpSampling2D((2,2),interpolation='bilinear')(input)

    output = Conv2D(filters=filter, kernel_size=kernel_size,kernel_initializer='he_normal', padding=padding)(input)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    return output


def segnet(image_feature):
    blockl_deconv1 = segnet_decoder_block(image_feature, filter=512,kernel_size=(3,3),upsample=True)
    block1_deconv2 = segnet_decoder_block(blockl_deconv1,filter=512)
    block1_deconv3 = segnet_decoder_block(block1_deconv2, filter=512)

    block2_deconv1 = segnet_decoder_block(block1_deconv3,filter=512,upsample=True)
    block2_deconv2 = segnet_decoder_block(block2_deconv1,filter=512)
    block2_deconv3 = segnet_decoder_block(block2_deconv2,filter=256)

    block3_deconv1 = segnet_decoder_block(block2_deconv3,filter=256,upsample=True)
    block3_deconv2 = segnet_decoder_block(block3_deconv1,filter=256)
    block3_deconv3 = segnet_decoder_block(block3_deconv2,filter=128)

    block4_deconv1 = segnet_decoder_block(block3_deconv3,filter=128,upsample=True)
    block4_deconv2 = segnet_decoder_block(block4_deconv1,filter=64)

    block5_deconv1 = segnet_decoder_block(block4_deconv2,filter=64,upsample=True)
    result = BatchNormalization()(Conv2D(filters=class_num,kernel_size=(1,1),padding='same')(block5_deconv1))

    return result

def unet_upblock(filter,input,concat,concat_encoder):
    up_conv = BatchNormalization()(Conv2D(filters=filter,kernel_size=(2,2),activation='relu',padding='same',
                                          kernel_initializer='he_normal')(UpSampling2D(size=(2,2),interpolation='bilinear')(input)))
    if concat:
        merge = concatenate([concat_encoder,up_conv])
    else:
        merge = up_conv
    conv1 = BatchNormalization()(Conv2D(filters=filter,kernel_size=(3,3),activation='relu',padding='same',
                                        kernel_initializer='he_normal')(merge))
    conv2 = BatchNormalization()(Conv2D(filters=filter, kernel_size=(3, 3), activation='relu', padding='same',
                                        kernel_initializer='he_normal')(conv1))

    return conv2


def unet(image_feature,vgg):
    conv1 = Conv2D(filters=1024,kernel_size=(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(image_feature)
    conv2 = Conv2D(filters=1024, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    decoder_input = Dropout(0.5)(conv2)

    decoder1 = unet_upblock(filter=512,input=decoder_input,concat_encoder=vgg.get_layer('block5_conv3').output,concat=True)
    decoder2 = unet_upblock(filter=256,input=decoder1,concat_encoder=vgg.get_layer('block4_conv3').output,concat=True)
    decoder3 = unet_upblock(filter=128,input=decoder2,concat_encoder=vgg.get_layer('block3_conv2').output,concat=False)
    decoder4 = unet_upblock(filter=64,input=decoder3,concat_encoder=vgg.get_layer('block2_conv2').output,concat=False)
    decoder5 = unet_upblock(filter=64, input=decoder4, concat_encoder=vgg.get_layer('block1_conv2').output,concat=False)

    result = Conv2D(filters=class_num,kernel_size=(1,1),activation='relu',padding='same',
                    kernel_initializer='he_normal',name='seg_output')(decoder5)

    return result


# def cross_entropy_loss(y_pred,y_true):
#     return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_pred,logits=y_true)

def softmax_sparse_crossentropy(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1])
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

vgg = VGG16(weights='imagenet',include_top = False,input_shape=(image_size,image_size,3))
image_feature = vgg.output
result = unet(image_feature,vgg)

model = Model(vgg.input, result)
model.summary()

model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),loss=softmax_sparse_crossentropy)#target_tensors=[output_target])

x_train = np.load('../data/seg/x_seg_train.npy')
y_train = np.load("../data/seg/y_seg_train.npy")
model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate,decay=learning_rate/epochs),loss=softmax_sparse_crossentropy)
model.fit(x=x_train,y=y_train,batch_size=16,epochs=epochs,validation_split=0.1,shuffle=True)