import math
import random
import tensorflow as tf
import tensorflow_addons as tfa

def applyRandomTransformation(X_input):
  X = tf.identity(X_input)
  X = tf.image.random_flip_left_right(X)
  X = tfa.image.transform_ops.rotate(X, random.random() * 2 * math.pi)
  X = tf.image.random_crop(X, size=[20, 20, 3])
  return X
