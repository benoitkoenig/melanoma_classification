total_train_rows = 33126
positives=584
negatives=32542

test_half_size = 24
batch_size = 10
epochs = 10

train_half_size = positives - test_half_size
steps_per_epoch = train_half_size * 2 // batch_size
