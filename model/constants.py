total_train_rows = 33126
positives=584
negatives=32542

train_half_size = 520
batch_size = 10

steps_per_epoch = 1
test_half_size = positives - train_half_size
epochs = train_half_size * 2 // batch_size