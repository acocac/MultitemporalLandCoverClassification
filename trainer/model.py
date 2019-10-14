# Copyright 2019 Google Inc. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines the model for product recommendation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow as tf
import sys
import os
import configparser
import functools
from tensorflow.python.lib.io import file_io

from constants.constants import *
from trainer import utils

from tensorflow.contrib.rnn import LSTMStateTuple

from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

## hyper parameters ##
tf.app.flags.DEFINE_string("kernel", "(1,3,3)", "kernel of convolutions")
tf.app.flags.DEFINE_string("classkernel", "(3,3)", "kernelsize of final classification convolution")
tf.app.flags.DEFINE_string("cnn_activation", "leaky_relu", "activation function for convolutional layers ('relu' or 'leaky_relu' [default])")

tf.app.flags.DEFINE_boolean("bidirectional", True, "Bidirectional Convolutional RNN")
tf.app.flags.DEFINE_integer("convrnn_compression_filters", -1, "number of convrnn compression filters or (default) -1 for no compression")
tf.app.flags.DEFINE_string("convcell", "gru", "Convolutional RNN cell architecture ('gru' (default) or 'lstm')")
tf.app.flags.DEFINE_string("convrnn_kernel", "(3,3)", "kernelsize of recurrent convolution. default (3,3)")
tf.app.flags.DEFINE_integer("convrnn_filters", 24, "number of convolutional filters in ConvLSTM/ConvGRU layer")
tf.app.flags.DEFINE_float("recurrent_dropout_i", 1., "input keep probability for recurrent dropout (default no dropout -> 1.)")
tf.app.flags.DEFINE_float("recurrent_dropout_c", 1., "state keep probability for recurrent dropout (default no dropout -> 1.)")
tf.app.flags.DEFINE_integer("convrnn_layers", 1, "number of convolutional recurrent layers")
tf.app.flags.DEFINE_boolean("peephole", False, "use peephole connections at convrnn layer. only for lstm (default False)")
tf.app.flags.DEFINE_boolean("convrnn_normalize", True, "normalize with batchnorm at convrnn layer (default True)")
tf.app.flags.DEFINE_string("aggr_strat", "state", "aggregation strategie to reduce temporal dimension (either default 'state' or 'sum_output' or 'avg_output')")

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Adam learning rate")
tf.app.flags.DEFINE_float("beta1", 0.9, "Adam beta1")
tf.app.flags.DEFINE_float("beta2", 0.999, "Adam beta2")
tf.app.flags.DEFINE_float("epsilon", 0.9, "Adam epsilon")

## expected data format ##
tf.app.flags.DEFINE_string("expected_datatypes",
                           "(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64)", "expected datatypes")
tf.app.flags.DEFINE_integer("pix10m", 24, "number of 10m pixels")
tf.app.flags.DEFINE_integer("num_bands_10m", 4, "number of bands in 10 meter resolution (4)")
tf.app.flags.DEFINE_integer("num_bands_20m", 6, "number of bands in 20 meter resolution (6)")
tf.app.flags.DEFINE_integer("num_bands_60m", 3, "number of bands in 20 meter resolution (3)")
tf.app.flags.DEFINE_integer("num_classes", 17, "number of classes not counting unknown class -> e.g. 0:uk,1:a,2:b,3:c,4:d -> num_classes 4")

## performance ##
tf.app.flags.DEFINE_boolean("swap_memory", True, "Swap memory between GPU and CPU for recurrent layers")

# #datadir
tf.app.flags.DEFINE_string('datadir', '', 'datadir')
tf.app.flags.DEFINE_string('modeldir', '', 'modeldir')
tf.app.flags.DEFINE_string('batchsize', '', 'batchsize')
tf.app.flags.DEFINE_string('train_on', '', 'train_on')
tf.app.flags.DEFINE_string('epochs', '', 'epochs')

FLAGS = tf.app.flags.FLAGS


def inference(input, is_train=True, num_classes=None):
    x, sequence_lengths = input

    rnn_output_list = list()
    rnn_state_list = list()

    x_rnn = x
    for j in range(1, FLAGS.convrnn_layers + 1):
        convrnn_kernel = eval(FLAGS.convrnn_kernel)
        x_rnn, state = convrnn_layer(input=x_rnn, is_train=is_train, filter=FLAGS.convrnn_filters,
                                     kernel=convrnn_kernel,
                                     bidirectional=FLAGS.bidirectional, convcell=FLAGS.convcell,
                                     sequence_lengths=sequence_lengths, scope="convrnn" + str(j))
        rnn_output_list.append(x_rnn)
        rnn_state_list.append(state)

    # # concat outputs from cnns and rnns in a dense scheme
    x = tf.concat(rnn_output_list, axis=-1)

    # # take concatenated states of last rnn block (might contain multiple conrnn layers)
    state = tf.concat(rnn_state_list, axis=-1)

    # use the cell state as featuremap for the classification step
    # cell state has dimensions (b,h,w,d) -> classification strategy
    if FLAGS.aggr_strat == 'state':

        class_input = state  # shape (b,h,w,d)
        classkernel = eval(FLAGS.classkernel)
        logits = conv_bn_relu(input=class_input, is_train=is_train, filter=num_classes,
                              kernel=classkernel, dilation_rate=(1, 1), conv_fun=tf.keras.layers.Conv2D,
                              var_scope="class")

    elif (FLAGS.aggr_strat == 'avg_output') or (FLAGS.aggr_strat == 'sum_output'):
        # last rnn output at each time t
        # class_input = x_rnn  # shape (b,t,h,w,d)
        class_input = x  # shape (b,t,h,w,d)

        # kernel = (1,FLAGS.classkernel[0],FLAGS.classkernel[1])
        kernel = (1, eval(FLAGS.classkernel)[0], eval(FLAGS.classkernel)[1])

        # logits for each single timeframe
        logits = conv_bn_relu(input=class_input, is_train=is_train, filter=num_classes, kernel=kernel,
                              dilation_rate=(1, 1, 1), conv_fun=tf.keras.layers.Conv3D, var_scope="class")

        if FLAGS.aggr_strat == 'avg_output':
            # average logit scores at each observation
            # (b,t,h,w,d) -> (b,h,w,d)
            logits = tf.reduce_mean(logits, axis=1)
        elif FLAGS.aggr_strat == 'sum_output':
            # summarize logit scores at each observation
            # the softmax normalization later will normalize logits again
            # (b,t,h,w,d) -> (b,h,w,d)
            logits = tf.reduce_sum(logits, axis=1)

    else:
        raise ValueError("please provide valid aggr_strat flag ('state' or 'avg_output' or 'sum_output')")

    tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, logits)

    return logits


def loss(logits, labels, mask, name):

    loss_per_px = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

    _ = tf.identity(loss_per_px, name="loss_per_px")
    _ = tf.identity(mask, name="mask_per_px")

    lpp = tf.boolean_mask(loss_per_px, mask)

    return tf.reduce_mean(lpp, name=name)

def optimize_old(loss, x250, name):
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    global_step = tf.train.get_global_step()

    samples_seen = tf.Variable(0, name='samples_seen', trainable=False)

    batchsize = tf.shape(x250)[0]
    samples_seen_increment_op = tf.compat.v1.assign(samples_seen, samples_seen + batchsize)

    lr = tf.compat.v1.placeholder_with_default(FLAGS.learning_rate, shape=(), name="learning_rate")
    beta1 = tf.compat.v1.placeholder_with_default(FLAGS.beta1, shape=(), name="beta1")
    beta2 = tf.compat.v1.placeholder_with_default(FLAGS.beta2, shape=(), name="beta2")

    samples_seen_increment_op2 = tf.identity(
        samples_seen_increment_op)  # <--- bug fixed to work with >TF1.4 this based on https://stackoverflow.com/questions/57385532/adamoptimizer-defies-tf-control-dependencies
    with tf.control_dependencies([samples_seen_increment_op2]):
        # with tf.control_dependencies([samples_seen_increment_op]): # execute this every time global step is incremented
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2,
                                                     epsilon=FLAGS.epsilon)
        return optimizer.minimize(loss, global_step=global_step, name=name)

def optimize(loss, name):
    global_step = tf.train.get_or_create_global_step()

    lr = tf.compat.v1.placeholder_with_default(FLAGS.learning_rate, shape=(), name="learning_rate")
    beta1 = tf.compat.v1.placeholder_with_default(FLAGS.beta1, shape=(), name="beta1")
    beta2 = tf.compat.v1.placeholder_with_default(FLAGS.beta2, shape=(), name="beta2")

    optimizer =  tf.compat.v1.train.AdamOptimizer(
        learning_rate=lr, beta1=beta1, beta2=beta2,
        epsilon=FLAGS.epsilon
    )

    up_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(up_ops):
        return optimizer.minimize(loss, global_step=global_step, name=name)

def metrics(logits, labels, mask):

    prediction_scores = tf.nn.softmax(logits=logits, name="prediction_scores")
    predictions = tf.argmax(prediction_scores, 3, name="predictions")

    targets = tf.argmax(labels, 3, name="targets")
    # correctly_predicted = tf.equal(predictions, targets, name="correctly_predicted")
    correctly_predicted = tf.equal(tf.boolean_mask(predictions, mask), tf.boolean_mask(targets, mask),
                                   name="correctly_predicted")
    overall_accuracy = tf.reduce_mean(tf.cast(correctly_predicted, tf.float32), name="overall_accuracy")

    overall_accuracy_sum = tf.Variable(tf.zeros(shape=([]), dtype=tf.float32),
                                       trainable=False,
                                       name="overall_accuracy_result",
                                       collections=[tf.GraphKeys.LOCAL_VARIABLES])

    update_op = tf.assign_add(overall_accuracy_sum, overall_accuracy)

    return(overall_accuracy, update_op)

def summary(metric_op, loss_op):
    """
    minimial summaries for training @ monitoring
    """

    tf.compat.v1.summary.scalar("accuracy", metric_op)
    tf.compat.v1.summary.scalar("loss", loss_op)
    ## histograms

    return tf.compat.v1.summary.merge_all()

def input(features, labels):

    with tf.name_scope("raw"):
        x10, x20, x60, doy, year = features

        x10 = tf.cast(x10, tf.float32, name="x10")
        x20 = tf.cast(x20, tf.float32, name="x20")
        x60 = tf.cast(x60, tf.float32, name="x60")
        doy = tf.cast(doy, tf.float32, name="doy")
        year = tf.cast(year, tf.float32, name="year")
        y = tf.cast(labels, tf.int32, name="y")

    # x10.set_shape([2,46,24,24,5])

    # integer sequence lenths per batch for dynamic_rnn masking
    seq = sequence_lengths = tf.reduce_sum(tf.cast(x10[:, :, 0, 0, 0] > 0, tf.int32), axis=1,
                                                name="sequence_lengths")

    def resize(tensor, new_height, new_width):
        b = tf.shape(tensor)[0]
        t = tf.shape(tensor)[1]
        h = tf.shape(tensor)[2]
        w = tf.shape(tensor)[3]
        d = tf.shape(tensor)[4]

        # stack batch on times to fit 4D requirement of resize_tensor
        stacked_tensor = tf.reshape(tensor, [b * t, h, w, d])
        reshaped_stacked_tensor = tf.image.resize_images(stacked_tensor, size=(new_height, new_width))
        return tf.reshape(reshaped_stacked_tensor, [b, t, new_height, new_width, d])

    def expand3x(vector):
        vector = tf.expand_dims(vector, -1)
        vector = tf.expand_dims(vector, -1)
        vector = tf.expand_dims(vector, -1)
        return vector

    with tf.name_scope("reshaped"):
        b = tf.shape(x10)[0]
        t = tf.shape(x10)[1]
        px = tf.shape(x10)[2]

        # b,t,w,h,d = x10.shape()

        x20 = tf.identity(resize(x20, px, px), name="x20")
        x60 = tf.identity(resize(x60, px, px), name="x60")

        tf.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, tf.identity(x10, name="x10"))
        tf.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x20)
        tf.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x60)

        # expand
        doymat = tf.multiply(expand3x(doy), tf.ones((b, t, px, px, 1)), name="doy")
        yearmat = tf.multiply(expand3x(year), tf.ones((b, t, px, px, 1)), name="year")

        x = tf.concat((x10, x20, x60, doymat, yearmat), axis=-1, name="x")
        tf.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x)

        # set depth of x for convolutions
        depth = FLAGS.num_bands_10m + FLAGS.num_bands_20m + FLAGS.num_bands_60m + 2  # doy and year

        # dynamic shapes. Fill for debugging

        x.set_shape([None, None, FLAGS.pix10m, FLAGS.pix10m, depth])
        y.set_shape((None, None, FLAGS.pix10m, FLAGS.pix10m))

    return (x, sequence_lengths), (y,)

def _model_fn(features, labels, mode, params):
  """MTLCC model

  Args:
    features: a batch of features.
    labels: a batch of labels or None if predicting.
    mode: an instance of tf.estimator.ModeKeys.
    params: a dict of additional params.

  Returns:
    A tf.estimator.EstimatorSpec that fully defines the model that will be run
      by an Estimator.
  """

  def parse(var, dtype):
      if type(var) == dtype:
          return var
      else:
          return eval(var)

  num_classes = parse(FLAGS.num_classes, int)

  # input pipeline
  with tf.name_scope("input"):
      (x, sequence_lengths), (alllabels,) = input(features, labels)
      alllabels = alllabels

  logits = inference(input=(x, sequence_lengths), num_classes=num_classes)

  # take first label -> assume labels do not change over timeseries
  first_labelmap = alllabels[:, 0]

  # create one-hot labelmap from 0-num_classes
  labels = tf.one_hot(first_labelmap, num_classes + 1)

  # mask out class 0 -> unknown
  unknown_mask = tf.cast(labels[:, :, :, 0], tf.bool)
  not_unknown_mask = tf.logical_not(unknown_mask)

  # keep rest of labels
  labels = labels[:, :, :, 1:]

  # Prediction op: Return the top k closest items for a given user or item.
  if mode == tf.estimator.ModeKeys.PREDICT:
    print('nothing yet')

  loss_op = loss(logits=logits, labels=labels, mask=not_unknown_mask, name="loss")

  metrics_ops = metrics(logits=logits, labels=labels, mask=not_unknown_mask)

  summary(metrics_ops[0], loss_op)

  if mode == tf.estimator.ModeKeys.TRAIN:
      print("building optimizer...")
      train_op = optimize(loss_op, name="train_op")
      logging_hook = tf.train.LoggingTensorHook({"accuracy": metrics_ops[0]}, every_n_iter=1)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op, training_hooks=[logging_hook])

  eval_metric_ops = {
            "accuracy": metrics_ops}

  modelpath = FLAGS.modeldir
  if modelpath is not None:
      # write FLAGS to file
      cfg = configparser.ConfigParser()
      flags_dict = dict()
      for name in FLAGS:
          flags_dict[name] = str(FLAGS[name].value)

      cfg["flags"] = flags_dict  # FLAGS.__flags #  broke tensorflow=1.5

      path = os.path.join(modelpath, MODEL_CFG_FILENAME)
      print("writing parameters to {}".format(path))
      with file_io.FileIO(path, 'w') as configfile:  # gcp
          # with open(path, 'w') as configfile:
          cfg.write(configfile)

      # write operation names to file for easier debugging
      path = os.path.join(modelpath, "nodes.txt")
      print("writing nodes list to {}".format(path))
      names = [n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node]
      with file_io.FileIO(path, 'w') as f:  # gcp
          # with open(path, 'w') as f:
          f.write("\n".join(names))

  return tf.estimator.EstimatorSpec(mode, loss=loss_op, eval_metric_ops=eval_metric_ops)
  # return tf.estimator.EstimatorSpec(mode, loss=loss_op)

def convrnn_layer(input, filter, is_train=True, kernel=FLAGS.convrnn_kernel, sequence_lengths=None, bidirectional=True,
                  convcell='gru', scope="convrnn"):
    with tf.compat.v1.variable_scope(scope):

        x = input

        px = x.get_shape()[3]

        if FLAGS.convrnn_compression_filters > 0:
            x = conv_bn_relu(input=x, is_train=is_train, filter=FLAGS.convrnn_compression_filters, kernel=(1, 1, 1),
                             dilation_rate=(1, 1, 1), var_scope="comp")

        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, tf.identity(input, "input"))

        if convcell == 'gru':
            cell = utils.ConvGRUCell((px, px), filter, kernel, activation=tf.nn.tanh,
                                       normalize=FLAGS.convrnn_normalize)

            # tf.Variable(tf.zeros((b,h,w)), validate_shape=False, trainable=False, name="zero_state")
            # zero_state = tf.Variable(tf.zeros((None, None, None), tf.float32), trainable=False)
        elif convcell == 'lstm':
            cell = utils.ConvLSTMCell((px, px), filter, kernel, activation=tf.nn.tanh,
                                        normalize=FLAGS.convrnn_normalize, peephole=FLAGS.peephole)
        else:
            raise ValueError("convcell argument {} not valid either 'gru' or 'lstm'".format(convcell))

        ## add dropout wrapper to cell
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=FLAGS.recurrent_dropout_i,
                                             state_keep_prob=FLAGS.recurrent_dropout_i)

        if bidirectional:
            outputs, final_states = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=x,
                                                                              sequence_length=sequence_lengths,
                                                                              dtype=tf.float32, time_major=False,
                                                                              swap_memory=FLAGS.swap_memory)

            concat_outputs = tf.concat(outputs, -1)

            if convcell == 'gru':
                concat_final_state = tf.concat(final_states, -1)
            elif convcell == 'lstm':
                fw_final, bw_final = final_states
                concat_final_state = LSTMStateTuple(
                    c=tf.concat((fw_final.c, bw_final.c), -1),
                    h=tf.concat((fw_final.h, bw_final.h), -1)
                )
                # else:
                #    concat_final_state = tf.concat((fw_final,bw_final),-1)
        else:
            concat_outputs, concat_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=input,
                                                                   sequence_length=sequence_lengths,
                                                                   dtype=tf.float32, time_major=False)

        if convcell == 'lstm':
            concat_final_state = concat_final_state.c

        else:
            tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME,
                                           tf.identity(concat_outputs, name="outputs"))
            tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME,
                                           tf.identity(concat_final_state, name="final_states"))

        return concat_outputs, concat_final_state


def conv_bn_relu(var_scope="name_scope", is_train=True, **kwargs):
    with tf.compat.v1.variable_scope(var_scope):

        if FLAGS.cnn_activation == 'relu':
            activation_function = tf.nn.relu
        elif FLAGS.cnn_activation == 'leaky_relu':
            activation_function = tf.nn.leaky_relu
        else:
            raise ValueError("please provide valid 'cnn_activation' FLAG. either 'relu' or 'leaky_relu'")

        is_train = tf.constant(is_train, dtype=tf.bool)
        x = conv_layer(**kwargs)
        x = Batch_Normalization(x, is_train, "bn")
        x = activation_function(x)

        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x)
        return x


def conv_layer(input, filter, kernel, dilation_rate=(1, 1, 1), stride=1, conv_fun=tf.keras.layers.Conv3D,
               layer_name="conv"):  # based on https://github.com/tensorflow/tensorflow/issues/26145

    with tf.name_scope(layer_name):
        # pad input to required sizes for same output dimensions
        input = pad(input, kernel, dilation_rate, padding="REFLECT")

        network = conv_fun(use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='VALID',
                           dilation_rate=dilation_rate)(inputs=input)

        return network


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def pad(input, kernel, dilation, padding="REFLECT"):
    """https://www.tensorflow.org/api_docs/python/tf/pad"""

    # determine required padding sizes
    def padsize(kernel, dilation):
        p = []
        for k, d in zip(kernel, dilation):
            p.append(int(int(k / 2) * d))
            # p.append(k/2)
        return p

    padsizes = padsize(kernel, dilation)

    # [bleft,bright], [tleft,tright], [hleft,hright], [wleft,wright],[dleft,dright]
    paddings = tf.constant([[0, 0]] + [[p, p] for p in padsizes] + [[0, 0]], dtype=tf.int32)

    return tf.pad(input, paddings, padding)


def _get_trial_id():
  """Returns the trial id if it exists, else "0"."""
  trial_id = json.loads(
      os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", "")
  return trial_id if trial_id else "1"


def build_estimator(run_config, params):
    """Returns TensorFlow estimator."""

    estimator = tf.estimator.Estimator(
        model_fn=_model_fn,
        model_dir=params.modeldir,
        config=run_config,
    )
    return estimator
