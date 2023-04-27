import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd

def get_data(data_dir):
    data = []
    path = os.path.join(data_dir)
    for img in os.listdir(path):
        if img.endswith('.png'):
            img = Image.open(os.path.join(path, img)).convert("L")
            img.load()
            img.thumbnail((64,64), Image.ANTIALIAS)
            array = np.asarray(img, dtype="int32")
            data.append(array)
    return np.array(list(data))

X = get_data("/kaggle/input/panda-resized-tt/train_images/train_images/train_images")
X_test = get_data('/kaggle/input/panda-resized-tt/test_images/test_images')

df = pd.read_csv("/kaggle/input/panda-resized-tt/train1.csv")
df['result'] = np.where(df['isup_grade'].astype(int)<=1, 0, 1)
Y = df['result'].astype(str).to_numpy()

df1 = pd.read_csv("/kaggle/input/panda-resized-tt/test.csv")
df1['result'] = np.where(df1['isup_grade'].astype(int)<=1, 0, 1)
Y_test = df1['result'].astype(str).to_numpy()

from tensorflow.keras.utils import to_categorical

X = np.expand_dims(X, axis=3)
Y = to_categorical(Y)
X_test = np.expand_dims(X_test, axis=3)
Y_test = to_categorical(Y_test)

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

num_output_classes = 2
input_img_size = (64, 64, 1)  # 64x64 image with 1 color channel

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_img_size))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_output_classes, activation="softmax"))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)

batch_size = 128
epochs = 15

history  = model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.15)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model_path ='/kaggle/working/modelB2.h5'
model.save(model_path)