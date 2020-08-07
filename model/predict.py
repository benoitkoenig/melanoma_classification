import csv

from melanoma_classification.model.data_generator.prediction_generator import getTestDataGenerator
from melanoma_classification.model.model import model

with open('data/test.csv') as csvFile:
  csvReader = csv.reader(csvFile, delimiter=',')
  rows = [r for r in csvReader][1:21]

def savePredictions(predictions):
  with open('submissions/latest.csv', 'w') as csvFile:
    csvWriter = csv.writer(csvFile, delimiter=',')
    for entry in zip(rows, predictions):
      csvWriter.writerow([entry[0][0], int(entry[1][1] > 0.95)])

model.load_weights('weights/weights')
dataGenerator = getTestDataGenerator(rows)
predictions = model.predict(dataGenerator)
savePredictions(predictions)
