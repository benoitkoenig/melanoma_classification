import csv
import pydicom
import numpy as np
import tensorflow as tf
import random

from melanoma_classification.model.constants import batch_size

def generateBalancedRows(rows):
  negatives = [r for r in rows if r[7] == "0"]
  positives = [r for r in rows if r[7] == "1"]
  random.shuffle(negatives)
  pickedNegatives = negatives[:len(positives)]
  balancedRows = positives + pickedNegatives
  random.shuffle(balancedRows)
  return balancedRows

def getXForAGivenRow(row):
  dicomData = pydicom.read_file('data/train/%s.dcm' % row[0])
  pixelArray = dicomData.pixel_array
  X = tf.convert_to_tensor(pixelArray)
  X = tf.image.resize(X, (224, 224))
  X = X.numpy()
  return X

def dataGenerator():
  with open('data/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    rows = [r for r in csv_reader][1:]
  rowsLeft = generateBalancedRows(rows)
  while len(rowsLeft) != 0:
    batch = rowsLeft[:batch_size]
    rowsLeft = rowsLeft[batch_size:]
    X = [getXForAGivenRow(row) for row in batch]
    targets = [int(row[7]) for row in batch]
    Y = [[1 - target, target] for target in targets]
    yield np.array(X), np.array(Y)
