# Multitemporal Land Cover Classification Network adapted to tf.Estimator and Google AI SDK

A recurrent neural network approach to encode multi-temporal data for land cover classification.

##### Source code of Rußwurm & Körner (2018) [PDF](https://arxiv.org/abs/1802.02080)

If you use this repository consider citing
```
Rußwurm M., Körner M. (2018). Multi-Temporal Land Cover Classification with
Sequential Recurrent Encoders. Arxiv, 2018.
```

<!--
```
Rußwurm M., Körner M. (2018). Multi-Temporal Land Cover Classification with
Sequential Recurrent Encoders. ISPRS International Journal of Geo-Information, 2018.
```
-->

The original `Tensorflow 1.7` repository located at `https://github.com/TUM-LMF/MTLCC` was adapted to TF 1.14 using tf.Estimator and Google AI structure.
Scripts for training and evaluation are provided.
The code can be executed after downloading the demo data (inside the repository folder).
After installing the dependencies the python scripts should be executable.

## Network

Similar to an encoding rnn layer of [sequence-to-sequence](https://www.tensorflow.org/tutorials/seq2seq) a variable-length input sequence of images is encoded to intermediate reprentation.

Encoding LSTM-RNN:
<p align="center">
<img src="doc/lstm.gif" width="500" />
</p>

Network structure
<p align="center">
  <img src="doc/network.png">
</p>
Bidirectional rnn encoder and convolutional softmax classifier, as described in the paper.

## Dependencies
Implementations of ConvGRU and ConvLSTM was adapted from https://github.com/carlthome/tensorflow-convlstm-cell and adapted into the trainer/utils.py script.

Python packages
```bash
conda install -y gdal
pip install tensorflow-gpu=1.14
pip install pandas
pip install configparser
pip install --upgrade google-api-python-client
```

## Download demo data

download demo data (requirement to run the following commands)

```bash
bash download.sh
```

## Network training and evaluation

### on local machine (requires dependencies installed)

#### train the network graph for 24px tiles
```bash
bash bin/run.train.local.sh
```

## Monitor training/validation curves

### on local machine (requires dependencies installed)

```bash
tensorboard --logdir=.
```
