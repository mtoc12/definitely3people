""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model



class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d1 = Dense(128, input_shape=(27,), activation='relu')
    self.d2 = Dense(64, activation='relu')
    self.d3 = Dense(32, activation='relu')
    self.d4 = Dense(16, activation='relu')
    self.dN = Dense(2)
    self.dr1 = Dropout(rate=0.1)
    self.dr2 = Dropout(rate=0.1)
    self.dr3 = Dropout(rate=0.1)

  def call(self, x):
    x = self.d1(x)
    x = self.dr1(x)
    x = self.d2(x)
    x = self.dr2(x)
    x = self.d3(x)
    x = self.dr3(x)
    x = self.d4(x)
    return self.dN(x)

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


# Import data
from numpy import genfromtxt
np_data = genfromtxt('train.csv', delimiter=',')
features = np_data[1:,:-1]
labels = np_data[1:,-1]

# test/train split
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=12)
for train_index, test_index in kf.split(features):
    train_features, test_features = features[train_index], features[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
    
    # normalize
    pass #?

    # Add a channels dimension
    train_ds = tf.expand_dims(train_ds, 0)
    test_ds = tf.expand_dims(test_ds, 0)


if False:
    # Import data
    from numpy import genfromtxt
    np_data = genfromtxt('train.csv', delimiter=',')
    whole_dataset = tf.data.Dataset.from_tensor_slices((np_data[:,:-1], np_data[:,-1]))

    # test/train split
    DATASET_SIZE = np_data.shape[0]
    train_size = int(0.7 * DATASET_SIZE)
    #val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.30 * DATASET_SIZE)

    #full_dataset = tf.data.TFRecordDataset(FLAGS.input_file)
    full_dataset = whole_dataset.shuffle(buffer_size=1000000)
    train_ds = full_dataset.take(train_size)
    #test_dataset = full_dataset.skip(train_size)
    #val_dataset = test_dataset.skip(val_size)
    test_ds = full_dataset.take(test_size)


    # Add a channels dimension
    train_ds = train_ds[..., tf.newaxis]
    test_ds = test_ds[..., tf.newaxis]

if False:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0


    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



# Run model
EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))
