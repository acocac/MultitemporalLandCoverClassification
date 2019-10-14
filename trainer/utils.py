"""Utility functions for model training."""

import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
import numpy as np
import sys

from constants.constants import *

class ConvLSTMCell(tf.compat.v1.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.

  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=True, peephole=True, data_format='channels_last', reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    if data_format == 'channels_last':
        self._size = tf.TensorShape((shape[0],shape[1],self._filters))
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape((self._filters, shape[0],shape[1]))
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    c, h = state

    x = tf.concat([x, h], axis=self._feature_axis)
    n = x.shape[-1].value
    m = 4 * self._filters if self._filters > 1 else 4
    W = tf.compat.v1.get_variable('kernel', tf.TensorShape((self._kernel[0],self._kernel[1],n,m)))
    y = convolution(x, W, data_format=self._data_format)
    if not self._normalize:
      y += tf.compat.v1.get_variable('bias', [m], initializer=tf.zeros_initializer())
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tf.compat.v1.get_variable('W_ci', c.shape[1:]) * c
      f += tf.compat.v1.get_variable('W_cf', c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      i = tf.contrib.layers.layer_norm(i)
      f = tf.contrib.layers.layer_norm(f)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tf.compat.v1.get_variable('W_co', c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state


class ConvGRUCell(tf.compat.v1.nn.rnn_cell.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""

  def __init__(self, shape, filters, kernel, activation=tf.tanh, normalize=True, data_format='channels_last', reuse=None):
    super(ConvGRUCell, self).__init__(_reuse=reuse)
    self._filters = filters
    self._kernel = kernel
    self._activation = activation
    self._normalize = normalize
    if data_format == 'channels_last':
        self._size = tf.TensorShape((shape[0], shape[1], self._filters))
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape((self._filters, shape[0], shape[1]))
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def call(self, x, h):
    channels = x.shape[self._feature_axis].value

    with tf.compat.v1.variable_scope('gates'):
      inputs = tf.concat([x, h], axis=self._feature_axis)
      n = channels + self._filters
      m = 2 * self._filters if self._filters > 1 else 2
      W = tf.compat.v1.get_variable('kernel', tf.TensorShape((self._kernel[0],self._kernel[1],n,m)))
      y = convolution(inputs, W, data_format=self._data_format)
      if self._normalize:
        r, u = tf.split(y, 2, axis=self._feature_axis)
        r = tf.contrib.layers.layer_norm(r)
        u = tf.contrib.layers.layer_norm(u)
      else:
        y += tf.compat.v1.get_variable('bias', [m], initializer=tf.ones_initializer())
        r, u = tf.split(y, 2, axis=self._feature_axis)
      r, u = tf.sigmoid(r), tf.sigmoid(u)

    with tf.compat.v1.variable_scope('candidate'):
      inputs = tf.concat([x, r * h], axis=self._feature_axis)
      n = channels + self._filters
      m = self._filters
      W = tf.compat.v1.get_variable('kernel', tf.TensorShape((self._kernel[0],self._kernel[1],n,m)))
      y = convolution(inputs, W, data_format=self._data_format)
      if self._normalize:
        y = tf.contrib.layers.layer_norm(y)
      else:
        y += tf.compat.v1.get_variable('bias', [m], initializer=tf.zeros_initializer())
      h = u * h + (1 - u) * self._activation(y)

    return h, h


class S2parser():
  """ defined the Sentinel 2 .tfrecord format """

  def __init__(self):

    self.feature_format = {
      'x10/data': tf.FixedLenFeature([], tf.string),
      'x10/shape': tf.FixedLenFeature([4], tf.int64),
      'x20/data': tf.FixedLenFeature([], tf.string),
      'x20/shape': tf.FixedLenFeature([4], tf.int64),
      'x60/data': tf.FixedLenFeature([], tf.string),
      'x60/shape': tf.FixedLenFeature([4], tf.int64),
      'dates/doy': tf.FixedLenFeature([], tf.string),
      'dates/year': tf.FixedLenFeature([], tf.string),
      'dates/shape': tf.FixedLenFeature([1], tf.int64),
      'labels/data': tf.FixedLenFeature([], tf.string),
      'labels/shape': tf.FixedLenFeature([3], tf.int64)
    }

    return None

  def write(self, filename, x10, x20, x60, doy, year, labels):
    # https://stackoverflow.com/questions/39524323/tf-sequenceexample-with-multidimensional-arrays

    writer = tf.python_io.TFRecordWriter(filename)

    x10 = x10.astype(np.int64)
    x20 = x20.astype(np.int64)
    x60 = x60.astype(np.int64)
    doy = doy.astype(np.int64)
    year = year.astype(np.int64)
    labels = labels.astype(np.int64)

    # Create a write feature
    feature = {
      'x10/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x10.tobytes()])),
      'x10/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x10.shape)),
      'x20/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x20.tobytes()])),
      'x20/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x20.shape)),
      'x60/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x60.tobytes()])),
      'x60/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x60.shape)),
      'labels/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tobytes()])),
      'labels/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=labels.shape)),
      'dates/doy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[doy.tobytes()])),
      'dates/year': tf.train.Feature(bytes_list=tf.train.BytesList(value=[year.tobytes()])),
      'dates/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=doy.shape))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

  def get_shapes(self, sample):
    print
    "reading shape of data using the sample " + sample
    data = self.read_and_return(sample)
    return [tensor.shape for tensor in data]

  def parse_example(self, serialized_example):
    """
    example proto can be obtained via
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    or by passing this function in dataset.map(.)
    """

    feature = tf.parse_single_example(serialized_example, self.feature_format)
    # decode and reshape x10
    x10 = tf.reshape(tf.decode_raw(feature['x10/data'], tf.int64), tf.cast(feature['x10/shape'], tf.int32))

    x20 = tf.reshape(tf.decode_raw(feature['x20/data'], tf.int64), tf.cast(feature['x20/shape'], tf.int32))
    x60 = tf.reshape(tf.decode_raw(feature['x60/data'], tf.int64), tf.cast(feature['x60/shape'], tf.int32))

    doy = tf.reshape(tf.decode_raw(feature['dates/doy'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))
    year = tf.reshape(tf.decode_raw(feature['dates/year'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))

    labels = tf.reshape(tf.decode_raw(feature['labels/data'], tf.int64), tf.cast(feature['labels/shape'], tf.int32))

    return x10, x20, x60, doy, year, labels

  def read(self, filenames):
    """ depricated! """

    if isinstance(filenames, list):
      filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    elif isinstance(filenames, tf.FIFOQueue):
      filename_queue = filenames
    else:
      print
      "please insert either list or tf.FIFOQueue"

    reader = tf.TFRecordReader()
    f, serialized_example = reader.read(filename_queue)

    print
    f

    feature = tf.parse_single_example(serialized_example, features=self.feature_format)

    # decode and reshape x10
    x10 = tf.reshape(tf.decode_raw(feature['x10/data'], tf.int64), tf.cast(feature['x10/shape'], tf.int32))

    x20 = tf.reshape(tf.decode_raw(feature['x20/data'], tf.int64), tf.cast(feature['x20/shape'], tf.int32))
    x60 = tf.reshape(tf.decode_raw(feature['x60/data'], tf.int64), tf.cast(feature['x60/shape'], tf.int32))

    doy = tf.reshape(tf.decode_raw(feature['dates/doy'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))
    year = tf.reshape(tf.decode_raw(feature['dates/year'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))

    labels = tf.reshape(tf.decode_raw(feature['labels/data'], tf.int64), tf.cast(feature['labels/shape'], tf.int32))

    return x10, x20, x60, doy, year, labels

  def tfrecord_to_pickle(self, tfrecordname, picklename):
    import cPickle as pickle

    reader = tf.TFRecordReader()

    # read serialized representation of *.tfrecord
    filename_queue = tf.train.string_input_producer([tfrecordname], num_epochs=None)
    filename_op, serialized_example = reader.read(filename_queue)
    feature = self.parse_example(serialized_example)

    with tf.Session() as sess:
      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      feature = sess.run(feature)

      coord.request_stop()
      coord.join(threads)

    pickle.dump(feature, open(picklename, "wb"), protocol=2)

  def read_and_return(self, filename):
    """ depricated! """

    # get feature operation containing
    feature_op = self.read([filename])

    with tf.Session() as sess:
      tf.global_variables_initializer()

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      return sess.run(feature_op)


def convolution(inputs,W,data_format):
    """wrapper around tf.nn.convolution with custom padding"""
    pad_h = int(int(W.get_shape()[0])/2)
    pad_w = int(int(W.get_shape()[1])/2)

    paddings = tf.constant([[0, 0], [pad_h,pad_h], [pad_w,pad_w], [0, 0]],dtype=tf.int32)

    inputs_padded = tf.pad(inputs, paddings, "REFLECT")

    return tf.nn.convolution(inputs_padded, W, 'VALID', data_format=data_format)
