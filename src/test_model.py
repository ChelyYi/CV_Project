import keras
import matplotlib
import matplotlib.pyplot as plot
import numpy as np
matplotlib.use("TkAgg")

model = keras.models.load_model("../model/classifier_pretrained.h5")

x_val = np.load("../data/x_val.npy")
y_val = np.load("../data/y_val.npy")

print(len(y_val))

k = 0
with open("record_prediction_pre",'w') as f:
    for i in range(len(x_val)):
        y_pred = model.predict(x_val[i].reshape(1,256,256,3))
        y_ = np.zeros(5)
        id_1 = np.argmax(y_pred)
        id_2 = np.where(y_val[i]==1)
        index = id_2[0]
        if id_1 not in index:
            k+= 1
            y_[id_1] = 1
            line = "ID: {} ; prediction: {}; ground truth:{} \n".format(i,y_,y_val[i])
            f.write(line)

print("error rate: {}".format(k/len(y_val)))





