total_train_rows = 33126
positives=584
negatives=32542

test_half_size = 104
batch_size = 4
epochs = 10 * 4

train_half_size = positives - test_half_size
steps_per_epoch = train_half_size * 2 // batch_size // 4
