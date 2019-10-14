# Copyright 2019 Google Inc. All Rights Reserved.
#
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
"""Main script to train the model for product recommendation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from constants.constants import *

from . import inputs
from . import model

import functools

def parse_arguments(argv):
  """Parses execution arguments and replaces default values.

  Args:
    argv: Input arguments from sys.

  Returns:
    Dictionary of parsed arguments.
  """

  parser = argparse.ArgumentParser()

  parser.add_argument('modeldir', type=str, help="directory containing TF graph definition 'graph.meta'")
  # parser.add_argument('--modelzoo', type=str, default="modelzoo", help='directory of model definitions (as referenced by flags.txt [model]). Defaults to environment variable $modelzoo')
  parser.add_argument('--datadir', type=str, default=None,
                      help='directory containing the data (defaults to environment variable $datadir)')
  parser.add_argument('-g', '--gpu', type=str, default="0", help='GPU')
  parser.add_argument('-d', '--train_on', type=str, default="2016", nargs='+',
                      help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
  parser.add_argument('-b', '--batchsize', type=int, default=16, help='batchsize')
  parser.add_argument('-v', '--verbose', action="store_true", help='verbosity')
  # parser.add_argument('-o', '--overwrite', action="store_true", help='overwrite graph. may lead to problems with checkpoints compatibility')
  parser.add_argument('-s', '--shuffle', type=bool, default=True, help="batch shuffling")
  parser.add_argument('-e', '--epochs', type=int, default=1, help="epochs")
  parser.add_argument('-t', '--temporal_samples', type=int, default=None, help="Temporal subsampling of dataset. "
                                                                               "Will at each get_next randomly choose "
                                                                               "<temporal_samples> elements from "
                                                                               "timestack. Defaults to None -> no temporal sampling")
  parser.add_argument('--save_frequency', type=int, default=64, help="save frequency")
  parser.add_argument('--summary_frequency', type=int, default=64, help="summary frequency")
  parser.add_argument('-f', '--fold', type=int, default=0, help="fold (requires train<fold>.ids)")
  parser.add_argument('--prefetch', type=int, default=6, help="prefetch batches")
  parser.add_argument('--max_models_to_keep', type=int, default=5, help="maximum number of models to keep")
  parser.add_argument('--save_every_n_hours', type=int, default=1, help="save checkpoint every n hours")
  parser.add_argument('--queue_capacity', type=int, default=256, help="Capacity of queue")
  parser.add_argument('--allow_growth', type=bool, default=False, help="Allow dynamic VRAM growth of TF")
  parser.add_argument('--limit_batches', type=int, default=-1,
                      help="artificially reduce number of batches to encourage overfitting (for debugging)")
  parser.add_argument('--learning_rate', type=float, default=None,
                      help="overwrite learning rate. Required placeholder named 'learning_rate' in model")
  parser.add_argument('--convrnn_filters', type=int, default=8,
                      help="number of convolutional filters in ConvLSTM/ConvGRU layer")
  parser.add_argument('--convrnn_layers', type=int, default=1,
                      help="number of convolutional recurrent layers")

  args, _ = parser.parse_known_args(args=argv[1:])
  return args

def train_and_evaluate(args):
    """Runs model training and evaluation using TF Estimator API."""

    num_samples = 0
    # if if num batches artificially reduced -> adapt sample size
    if args.limit_batches > 0:
        num_samples = args.limit_batches * args.batchsize
    else:
        num_samples += int(inputs.input_filenames(args, mode=tf.estimator.ModeKeys.TRAIN).get_shape()[0])

    train_steps = num_samples / args.batchsize * args.epochs

    train_input_fn = functools.partial(
        inputs.input_fn,
        args,
        mode=tf.estimator.ModeKeys.TRAIN
    )

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=train_steps)

    eval_input_fn = functools.partial(
        inputs.input_fn,
        args,
        mode=tf.estimator.ModeKeys.EVAL
    )
    #
    # exporter = tf.estimator.FinalExporter(
    #     'export', functools.partial(
    #         inputs.tfrecord_serving_input_fn,
    #         feature_spec=feature_spec,
    #         label_name='labels'))
    #
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        start_delay_secs=1,
        throttle_secs=1,  # eval no more than every x seconds
        steps=1, # evals on x batches
        name=TESTING_SUMMARY_FOLDER_NAME
    )
    #
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=args.save_frequency,
        save_summary_steps=args.summary_frequency,
        keep_checkpoint_max=args.max_models_to_keep,
        keep_checkpoint_every_n_hours=args.save_every_n_hours,
        model_dir=args.modeldir,
        log_step_count_steps=args.summary_frequency # set the frequency of logging steps for loss function
    )

    estimator = model.build_estimator(
        run_config, args)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def main():
  args = parse_arguments(sys.argv)
  tf.logging.set_verbosity(tf.logging.INFO)
  train_and_evaluate(args)

if __name__ == "__main__":
  main()