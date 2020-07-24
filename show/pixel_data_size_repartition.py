#!/usr/bin/python

import os
import pydicom

filenames = [f for f in os.listdir('data/train') if f != ".gitignore"]
paths = [os.path.join('data/train/', f) for f in filenames]
sizes = {}
for path in paths:
  dicomData = pydicom.read_file(path)
  pixelArray = dicomData.pixel_array
  size = '%sx%s' % (len(pixelArray), len(pixelArray[0]))
  if size in sizes:
    sizes[size] += 1
  else:
    sizes[size] = 0
  print(sizes)
