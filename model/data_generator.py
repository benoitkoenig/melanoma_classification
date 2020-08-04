import csv
import pydicom
import numpy as np
import tensorflow as tf

with open('data/train.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  rows = [r for r in csv_reader][1:]

def dataGenerator():
  for row in rows:
    dicomData = pydicom.read_file('data/train/%s.dcm' % row[0])
    pixelArray = dicomData.pixel_array
    X = [tf.convert_to_tensor(pixelArray)]
    X = tf.image.resize(X, (223, 223))
    X = np.array(X)
    Y = np.array([int(row[7])])
    yield X, Y
