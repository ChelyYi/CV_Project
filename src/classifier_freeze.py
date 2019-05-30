import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense,Flatten

x_train = np.load("../data/x_train.npy")
y_train = np.load("../data/y_train.npy")

model = keras.models.load_model('../model/CNNautoencoder.h5')

input = model.input
feature_output = model.get_layer(name='Encoder2_Maxpooling').output
classification_result = Dense(units=5,activation='softmax',use_bias=False)(Flatten()(feature_output))

encoder = Model(input,classification_result)
for layer in encoder.layers[:-1]:
    layer.trainable = False

encoder.summary()
encoder.compile(loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])
encoder.fit(x_train,y_train,batch_size=32,epochs=30,validation_split=0.2)