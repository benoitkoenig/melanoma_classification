from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model

model = Sequential()
model.add(Input(shape=(224, 224, 3)))
model.add(VGG16(include_top=False, weights="imagenet", classes=2))

model.compile(optimizer="adam", loss="binary_crossentropy")
