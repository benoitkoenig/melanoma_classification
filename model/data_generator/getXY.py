import pydicom
import tensorflow as tf

def getX(row, folder='train'):
  dicomData = pydicom.read_file('data/%s/%s.dcm' % (folder, row[0]))
  pixelArray = dicomData.pixel_array
  X = tf.convert_to_tensor(pixelArray)
  X = tf.image.resize(X, (224, 224))
  return X

def getY(row):
  target = int(row[7])
  Y = [1 - target, target]
  return Y

def getXY(rows):
  X = tf.convert_to_tensor([getX(row) for row in rows])
  Y = tf.convert_to_tensor([getY(row) for row in rows])
  return X, Y
