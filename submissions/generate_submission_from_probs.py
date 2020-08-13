import csv

threshold = 0.7

with open('submissions/latest_probs.csv') as csvFile:
  csvReader = csv.reader(csvFile, delimiter=',')
  rows = [r for r in csvReader]

newRows = [[row[0], int(float(row[1]) > threshold)] for row in rows]

with open('submissions/latest.csv', 'w') as csvFile:
  csvWriter = csv.writer(csvFile, delimiter=',')
  csvWriter.writerow(['image_name', 'target'])
  for row in newRows:
    csvWriter.writerow(row)
