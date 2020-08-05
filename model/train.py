from melanoma_classification.model.model import model
from melanoma_classification.model.data_generator import dataGenerator
from melanoma_classification.model.constants import steps_per_epoch, epochs

gen = dataGenerator()
model.fit(gen, epochs=epochs, steps_per_epoch=steps_per_epoch)
