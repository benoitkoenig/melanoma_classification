from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

input = Input(shape=(224, 224, 3), name='image_input')

x = VGG16(weights='imagenet', include_top=False)(input)
x = Flatten(name='flatten')(x)
x = Dense(256, activation='relu', name='fc1')(x)
output = Dense(2, activation='softmax', name='predictions')(x)

model = Model(inputs=input, outputs=output)
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy")
