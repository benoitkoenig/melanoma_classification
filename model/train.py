from melanoma_classification.model.model import model
from melanoma_classification.model.data_generator import dataGenerator

gen = dataGenerator()
model.fit(gen, epochs=12)
