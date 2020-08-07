import tensorflow as tf

from melanoma_classification.model.constants import batch_size
from melanoma_classification.model.data_generator.getXY import getX

def getTestDataGenerator(rows):
  rowsLeft = rows
  while (len(rowsLeft) != 0):
    batch = rowsLeft[:batch_size]
    rowsLeft = rowsLeft[batch_size:]
    X = tf.convert_to_tensor([getX(row, folder='test') for row in batch])
    yield X
