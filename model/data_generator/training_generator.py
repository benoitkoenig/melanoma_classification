import csv
import pydicom
import tensorflow as tf
import random

from melanoma_classification.model.constants import batch_size, test_half_size
from melanoma_classification.model.data_generator.getXY import getXY

def generateBalancedRows(negatives, positives):
  negativesCopy = [row for row in negatives]
  random.shuffle(negativesCopy)
  pickedNegatives = negatives[:len(positives)]
  balancedRows = positives + pickedNegatives
  random.shuffle(balancedRows)
  return balancedRows

class DataGenerator:
  def __init__(self):
    with open('data/train.csv') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      rows = [r for r in csv_reader][1:]
    random.shuffle(rows)
    negatives = [r for r in rows if r[7] == '0']
    positives = [r for r in rows if r[7] == '1']

    self.positivesTest = positives[:test_half_size]
    self.positivesTrain = positives[test_half_size:]
    self.negativesTest = negatives[:test_half_size]
    self.negativesTrain = negatives[test_half_size:]

  def getValidationSet(self):
    rows = self.positivesTest + self.negativesTest
    random.shuffle(rows)
    X, Y = getXY(rows)
    return X, Y

  def getTrainingDataGenerator(self):
    rowsLeft = generateBalancedRows(self.negativesTrain, self.positivesTrain)
    while len(rowsLeft) != 0:
      X, Y = getXY(rowsLeft[:batch_size])
      rowsLeft = rowsLeft[batch_size:]
      yield X, Y
