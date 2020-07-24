#!/usr/bin/python

import sys
import csv
import pydicom
import matplotlib.pyplot as plt

def getEntryName():
  if (len(sys.argv) < 2):
    return 'ISIC_2637011'
  param = sys.argv[1]
  if (param[:4] == 'ISIC'):
    return param
  index = int(param)
  with open('data/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    rows = [r for r in csv_reader]
    name = rows[index + 1][0]
    return name

def getCsvData(name):
  with open('data/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    rows = [r for r in csv_reader if r[0] == name]
  row = rows[0]
  return row

def getDicomData(name):
  dicomData = pydicom.read_file('data/train/%s.dcm' % name)
  return dicomData

def displayImage(pixelData):
  fig, ax = plt.subplots()
  image = ax.imshow(pixelData)
  plt.show()

name = getEntryName()
csvData = getCsvData(name)
dicomData = getDicomData(name)

print('%s\n\n%s' % (csvData, dicomData))
displayImage(dicomData.pixel_array)
