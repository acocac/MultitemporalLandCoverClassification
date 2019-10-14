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
"""Input and preprocessing functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
import os
import configparser
import csv
import numpy as np
from tensorflow.python.lib.io import file_io
# from google.cloud import storage
import re

from trainer import utils
from constants.constants import *


def input_fn(args, mode):
  """Reads TFRecords and returns the features and labels."""

  dataset = Dataset(datadir=args.datadir, verbose=True, temporal_samples=args.temporal_samples, section=args.train_on[0])

  if mode == tf.estimator.ModeKeys.TRAIN:
    partition = TRAINING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.EVAL:
    partition = TESTING_IDS_IDENTIFIER
  else:
    partition = EVAL_IDS_IDENTIFIER

  # datasets_dict[section][partition] = dict()
  tfdataset, _, _, filenames = dataset.create_tf_dataset(partition,
                                                         args.fold,
                                                         args.batchsize,
                                                         args.shuffle,
                                                         prefetch_batches=args.prefetch,
                                                         num_batches=args.limit_batches)

  # iterator = tfdataset.dataset.make_one_shot_iterator()
  iterator = tfdataset.make_initializable_iterator()

  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

  features, labels = iterator.get_next()

  if mode == tf.estimator.ModeKeys.PREDICT:
    return features

  return features, labels

def input_filenames(args, mode):
  """Reads TFRecords and returns the features and labels."""

  dataset = Dataset(datadir=args.datadir, verbose=True, temporal_samples=args.temporal_samples, section=args.train_on[0])

  if mode == tf.estimator.ModeKeys.TRAIN:
    partition = TRAINING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.EVAL:
    partition = TESTING_IDS_IDENTIFIER
  else:
    partition = EVAL_IDS_IDENTIFIER

  # datasets_dict[section][partition] = dict()
  tfdataset, _, _, filenames = dataset.create_tf_dataset(partition,
                                                         args.fold,
                                                         args.batchsize,
                                                         args.shuffle,
                                                         prefetch_batches=args.prefetch,
                                                         num_batches=args.limit_batches)

  return filenames

class Dataset():
    """ A wrapper class around Tensorflow Dataset api handling data normalization and augmentation """

    def __init__(self, datadir, verbose=False, temporal_samples=None, section="dataset", augment=False):
      self.verbose = verbose

      self.augment = augment

      # parser reads serialized tfrecords file and creates a feature object
      parser = utils.S2parser()
      self.parsing_function = parser.parse_example

      self.temp_samples = temporal_samples
      self.section = section

      # if datadir is None:
      #    dataroot=os.environ["datadir"]
      # else:
      dataroot = datadir

      # csv list of geotransforms of each tile: tileid, xmin, xres, 0, ymax, 0, -yres, srid
      # use querygeotransform.py or querygeotransforms.sh to generate csv
      # fills dictionary:
      # geotransforms[<tileid>] = (xmin, xres, 0, ymax, 0, -yres)
      # srid[<tileid>] = srid
      self.geotransforms = dict()
      # https://en.wikipedia.org/wiki/Spatial_reference_system#Identifier
      self.srids = dict()
      with file_io.FileIO(os.path.join(dataroot, "geotransforms.csv"), 'r') as f:  # gcp
        # with open(os.path.join(dataroot, "geotransforms.csv"),'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
          # float(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5]), int(row[6]))
          self.geotransforms[str(row[0])] = (
            float(row[1]), float(row[2]), int(row[3]), float(row[4]), int(row[5]), float(row[6]))
          self.srids[str(row[0])] = int(row[7])

      classes = os.path.join(dataroot, "classes.txt")
      with file_io.FileIO(classes, 'r') as f:  # gcp
        # with open(classes, 'r') as f:
        classes = f.readlines()

      self.ids = list()
      self.classes = list()
      for row in classes:
        row = row.replace("\n", "")
        if '|' in row:
          id, cl = row.split('|')
          self.ids.append(int(id))
          self.classes.append(cl)

      ## create a lookup table to map labelids to dimension ids

      # map data ids [0, 2, 4,..., nclasses_originalID]
      labids = tf.constant(self.ids, dtype=tf.int64)

      # to dimensions [0, 1, 2, ... nclasses_orderID]
      dimids = tf.constant(list(range(0, len(self.ids), 1)), dtype=tf.int64)

      self.id_lookup_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(labids, dimids),
                                                         default_value=-1)

      self.inverse_id_lookup_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(dimids, labids),
        default_value=-1)

      # self.classes = [cl.replace("\n","") for cl in f.readlines()]

      cfgpath = os.path.join(dataroot, "dataset.ini")
      # load dataset configs
      datacfg = configparser.ConfigParser()
      with file_io.FileIO(cfgpath, 'r') as f:  # gcp
        datacfg.readfp(f)
      cfg = datacfg[section]

      self.tileidfolder = os.path.join(dataroot, "tileids")
      self.datadir = os.path.join(dataroot, cfg["datadir"])

      assert 'pix10' in cfg.keys()
      assert 'nobs' in cfg.keys()
      assert 'nbands10' in cfg.keys()
      assert 'nbands20' in cfg.keys()
      assert 'nbands60' in cfg.keys()

      self.tiletable = cfg["tiletable"]

      self.nobs = int(cfg["nobs"])

      self.expected_shapes = self.calc_expected_shapes(int(cfg["pix10"]),
                                                       int(cfg["nobs"]),
                                                       int(cfg["nbands10"]),
                                                       int(cfg["nbands20"]),
                                                       int(cfg["nbands60"])
                                                       )

      # expected datatypes as read from disk
      self.expected_datatypes = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64)

    def calc_expected_shapes(self, pix10, nobs, bands10, bands20, bands60):
      pix20 = pix10 / 2;
      pix60 = pix10 / 6;
      x10shape = (nobs, pix10, pix10, bands10)
      x20shape = (nobs, pix20, pix20, bands20)
      x60shape = (nobs, pix60, pix60, bands60)
      doyshape = (nobs,)
      yearshape = (nobs,)
      labelshape = (nobs, pix10, pix10)

      return [x10shape, x20shape, x60shape, doyshape, yearshape, labelshape]

    def transform_labels(self, feature):
      """
      1. take only first labelmap, as labels are not supposed to change
      2. perform label lookup as stored label ids might be not sequential labelid:[0,3,4] -> dimid:[0,1,2]
      """

      x10, x20, x60, doy, year, labels = feature

      # take first label time [46,24,24] -> [24,24]
      # labels are not supposed to change over the time series
      # labels = labels[0]
      labels = self.id_lookup_table.lookup(labels)

      return (x10, x20, x60, doy, year), labels

    def normalize(self, feature):

      x10, x20, x60, doy, year, labels = feature
      x10 = tf.scalar_mul(1e-4, tf.cast(x10, tf.float32))
      x20 = tf.scalar_mul(1e-4, tf.cast(x20, tf.float32))
      x60 = tf.scalar_mul(1e-4, tf.cast(x60, tf.float32))

      doy = tf.cast(doy, tf.float32) / 365

      # year = (2016 - tf.cast(year, tf.float32)) / 2017
      year = tf.cast(year, tf.float32) - 2016

      return x10, x20, x60, doy, year, labels

    def augment(self, feature):

      x10, x20, x60, doy, year, labels = feature

      ## Flip UD

      # roll the dice
      condition = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)

      # flip
      x10 = tf.cond(condition, lambda: tf.reverse(x10, axis=[1]), lambda: x10)
      x20 = tf.cond(condition, lambda: tf.reverse(x20, axis=[1]), lambda: x20)
      x60 = tf.cond(condition, lambda: tf.reverse(x60, axis=[1]), lambda: x60)
      labels = tf.cond(condition, lambda: tf.reverse(labels, axis=[1]), lambda: labels)

      ## Flip LR

      # roll the dice
      condition = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)

      # flip
      x10 = tf.cond(condition, lambda: tf.reverse(x10, axis=[2]), lambda: x10)
      x20 = tf.cond(condition, lambda: tf.reverse(x20, axis=[2]), lambda: x20)
      x60 = tf.cond(condition, lambda: tf.reverse(x60, axis=[2]), lambda: x60)
      labels = tf.cond(condition, lambda: tf.reverse(labels, axis=[2]), lambda: labels)

      return x10, x20, x60, doy, year, labels

    def temporal_sample(self, feature):
      """ randomy choose <self.temp_samples> elements from temporal sequence """

      n = self.temp_samples

      # skip if not specified
      if n is None:
        return feature

      x10, x20, x60, doy, year, labels = feature

      # data format 1, 2, 1, 2, -1,-1,-1
      # sequence lengths indexes are negative values.
      # sequence_lengths = tf.reduce_sum(tf.cast(x10[:, :, 0, 0, 0] > 0, tf.int32), axis=1)

      # tf.sequence_mask(sequence_lengths, n_obs)

      # max_obs = tf.shape(x10)[1]
      max_obs = self.nobs

      shuffled_range = tf.random_shuffle(tf.range(max_obs))[0:n]

      idxs = -tf.nn.top_k(-shuffled_range, k=n).values

      x10 = tf.gather(x10, idxs)
      x20 = tf.gather(x20, idxs)
      x60 = tf.gather(x60, idxs)
      doy = tf.gather(doy, idxs)
      year = tf.gather(year, idxs)

      return x10, x20, x60, doy, year, labels

    def get_ids(self, partition, fold=0):

      def readids(path):
        with open(path, 'r') as f:
          lines = f.readlines()
        return [int(l.replace("\n", "")) for l in lines]

      traintest = "{partition}_fold{fold}.tileids"
      eval = "{partition}.tileids"

      if partition == 'train':
        # e.g. train240_fold0.tileids
        path = os.path.join(self.tileidfolder, traintest.format(partition=partition, fold=fold))
        return readids(path)
      elif partition == 'test':
        # e.g. test240_fold0.tileids
        path = os.path.join(self.tileidfolder, traintest.format(partition=partition, fold=fold))
        return readids(path)
      elif partition == 'eval':
        # e.g. eval240.tileids
        path = os.path.join(self.tileidfolder, eval.format(partition=partition))
        return readids(path)
      else:
        raise ValueError("please provide valid partition (train|test|eval)")

    def create_tf_dataset(self, partition, fold, batchsize, shuffle, prefetch_batches=None, num_batches=-1, threads=8,
                          drop_remainder=False, overwrite_ids=None):

      # set of ids as present in database of given partition (train/test/eval) and fold (0-9)
      allids = self.get_ids(partition=partition, fold=fold)

      # set of ids present in local folder (e.g. 1.tfrecord)
      tiles = os.listdir(self.datadir)

      if tiles[0].endswith(".gz"):
        compression = "GZIP"
        ext = ".tfrecord.gz"
      else:
        compression = ""
        ext = ".tfrecord"

      downloaded_ids = [int(t.replace(".gz", "").replace(".tfrecord", "")) for t in tiles]

      # intersection of available ids and partition ods
      if overwrite_ids is None:
        ids = list(set(downloaded_ids).intersection(allids))
      else:
        print("overwriting data ids! due to manual input")
        ids = overwrite_ids

      filenames = [os.path.join(self.datadir, str(id) + ext) for id in ids]

      if self.verbose:
        print(
        "dataset: {}, partition: {}, fold:{} {}/{} tiles downloaded ({:.2f} %)".format(self.section, partition, fold,
                                                                                       len(ids), len(allids),
                                                                                       len(ids) / float(
                                                                                         len(allids)) * 100))

      def mapping_function(serialized_feature):
        # read data from .tfrecords
        feature = self.parsing_function(serialized_example=serialized_feature)
        # sample n times out of the timeseries
        feature = self.temporal_sample(feature)
        # perform data normalization [0,1000] -> [0,1]
        feature = self.normalize(feature)
        # perform data augmentation
        if self.augment: feature = self.augment(feature)
        # replace potentially non sequential labelids with sequential dimension ids
        feature = self.transform_labels(feature)
        return feature

      if num_batches > 0:
        filenames = filenames[0:num_batches * batchsize]

      # shuffle sequence of filenames
      if shuffle and partition == 'train':
        filenames = tf.random.shuffle(filenames)

      dataset = tf.data.TFRecordDataset(filenames, compression_type=compression)

      dataset = dataset.map(mapping_function, num_parallel_calls=threads)

      # repeat forever until externally stopped
      dataset = dataset.repeat()

      # Don't trust buffer size -> manual shuffle beforehand
      # if shuffle:
      #    dataset = dataset.shuffle(buffer_size=int(min_after_dequeue))

      if drop_remainder:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(int(batchsize)))
      else:
        dataset = dataset.batch(int(batchsize))

      if prefetch_batches is not None:
        dataset = dataset.prefetch(prefetch_batches)

      # assign output_shape to dataset

      # modelshapes are expected shapes of the data stacked as batch
      output_shape = []
      for shape in self.expected_shapes:
        output_shape.append(tf.TensorShape((batchsize,) + shape))

      return dataset, output_shape, self.expected_datatypes, filenames