import keras
import numpy as np

model_path = "../model/good_model_version_3_depth_3.h5"
data_path= "../data/x_val.npy"

model = keras.models.load_model(model_path)

x_val = np.load(data_path)

print(model.evaluate(x_val,x_val,batch_size=32))