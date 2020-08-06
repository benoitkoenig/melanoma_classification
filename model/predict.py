import csv
import math
import pydicom
import tensorflow as tf

from melanoma_classification.model.constants import batch_size
from melanoma_classification.model.model import model

with open('data/test.csv') as csvFile:
  csvReader = csv.reader(csvFile, delimiter=',')
  rows = [r for r in csvReader][1:21]

def getX(row):
  dicomData = pydicom.read_file('data/test/%s.dcm' % row[0])
  pixelArray = dicomData.pixel_array
  X = tf.convert_to_tensor(pixelArray)
  X = tf.image.resize(X, (224, 224))
  return X

def getTestDataGenerator():
  rowsLeft = rows
  while (len(rowsLeft) != 0):
    batch = rowsLeft[:batch_size]
    rowsLeft = rowsLeft[batch_size:]
    X = tf.convert_to_tensor([getX(row) for row in batch])
    yield X

def savePredictions(predictions):
  with open('submissions/latest.csv', 'w') as csvFile:
    csvWriter = csv.writer(csvFile, delimiter=',')
    for entry in zip(rows, predictions):
      csvWriter.writerow([entry[0][0], int(entry[1][1] > 0.5)])

model.load_weights('weights/weights')
dataGenerator = getTestDataGenerator()
predictions = model.predict(dataGenerator)
savePredictions(predictions)
