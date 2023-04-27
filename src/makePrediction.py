from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('modelB.h5')
model1 = load_model('modelisup2.h5')


def predict_binary_classification(path):
    img = Image.open(path).convert("L")
    img.load()
    img.thumbnail((64, 64), Image.ANTIALIAS)
    mi = np.asarray(img, dtype="int32")
    mi = np.expand_dims(mi, axis=0)
    mi = np.expand_dims(mi, axis=3)
    prediction = model.predict(mi)
    print([x for x in prediction])
    return prediction


def predict_isup(path):
    img = Image.open(path).convert("L")
    img.load()
    img.thumbnail((64, 64), Image.ANTIALIAS)
    mi = np.asarray(img, dtype="int32")
    mi = np.expand_dims(mi, axis=0)
    mi = np.expand_dims(mi, axis=3)
    prediction = model1.predict(mi)
    print([x for x in prediction])
    return prediction

#(predict_binary_classification("pandaneg.png"))
# (predict_binary_classification("transf.png"))
# (predict_binary_classification("pandaPos.png"))
# (predict_binary_classification("3da9.png"))
