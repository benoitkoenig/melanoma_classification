from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from melanoma_classification.model.model import model
from melanoma_classification.model.data_generator.training_generator import DataGenerator
from melanoma_classification.model.constants import steps_per_epoch, epochs, batch_size

dataGenerator = DataGenerator()
validation_data = dataGenerator.getValidationSet()
gen = dataGenerator.getTrainingDataGenerator()

csv_logger = CSVLogger('logs/training.csv')
model_checkpoint = ModelCheckpoint(
  filepath='weights/weights',
  save_weights_only=True,
  monitor='val_loss',
  mode='min',
  save_best_only=True,
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

model.fit(
  gen,
  epochs=epochs,
  steps_per_epoch=steps_per_epoch,
  validation_data=validation_data,
  validation_batch_size=batch_size,
  callbacks=[csv_logger, model_checkpoint, reduce_lr],
)
