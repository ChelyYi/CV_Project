import keras
import matplotlib
import matplotlib.pyplot as plot
import numpy as np
matplotlib.use("TkAgg")

model = keras.models.load_model("../model/CNNautoencoder.h5")

x_train = np.load("../data/x_train.npy")
sample = x_train[1007]
plot.imshow(sample)
plot.show()

y = model.predict(sample.reshape(1,256,256,3))


plot.imshow(y.reshape(256,256,3))
plot.show()
