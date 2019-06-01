import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense,Flatten,Input
from CNNautoencoder import encoder

x_train = np.load("../data/x_train.npy")
y_train = np.load("../data/y_train.npy")


def classifier(model_path,num_of_layers=2,filter_num_list=[16,3],is_fine_tuned=False):

    if model_path:
        model = keras.models.load_model(model_path)
        input =model.input
        enc_out = model.get_layer(name="Encoder2_Maxpooling").output

    else:
        input = Input(shape=(256,256,3))
        enc_out = encoder(num_of_layers,filter_num_list,input)

    enc_out = Flatten()(enc_out)
    classification_layer = Dense(units=5,activation="softmax",use_bias=False)(enc_out)
    classifier=Model(input,classification_layer)
    for layer in classifier.layers[:-1]:
        layer.trainable = is_fine_tuned

    return classifier


image_classifier = classifier(model_path='../model/CNNautoencoder.h5',is_fine_tuned=True)
image_classifier.summary()