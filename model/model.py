from tensorflow.python.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Lambda
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Lambda(preprocess_input, name="preprocess", input_shape=(224, 224, 3)))
model.add(EfficientNetB4(weights='imagenet', include_top=False))
model.add(Flatten(name='flatten'))
model.add(Dense(256, activation='relu', name='relu1'))
model.add(Dense(2, activation='softmax', name='predictions'))

optimizer = Adam(learning_rate=1e-5)

model.compile(optimizer=optimizer, loss='bce', metrics=['accuracy'])
