##THIS WAS IMPORTED FROM ADV. DATA SCIENCE
from pprint import pprint
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import callbacks
#from sklearn.model_selection import ParameterGrid #Just just tensorflow hparams or just a for loop=[-]
from tensorflow.keras.layers import Input, BatchNormalization, Activation
### This is from feedforward


import tensorflow as tf
import tensorflow_datasets as tfds
import time
import wrapt
import logging
import argparse

#Allows for logging to happen.
logging.basicConfig()
log = logging.getLogger("common")
log.setLevel(logging.INFO)


#tf.debugging.set_log_device_placement(True)

def parse_flags():
  parser = argparse.ArgumentParser()
  parser.add_argument("--normalize_input", action="store_true", help="normalize input data")
  parser.add_argument("--num_epochs", default=20, type=int, help="Number of epochs for each model to train")
  parser.add_argument("--num_trials", default=1, type=int, help="Number of trials overall")
  parser.add_argument("--num_dense_layers", default=1, type=int, help="Number of dense layers to add")
  parser.add_argument("--num_dense_units", default=128, type=int, help="Number of units in each dense layer")
  parser.add_argument("--gpu_train", action="store_true")
  parser.add_argument("--gpu_test", action="store_true")
  # todo : what parameters would we want to change?
  # helpful link: https://stackoverflow.com/a/15753721
  
  args = parser.parse_args()
  return args

#This is the logging algorithm given to me from Dr. Ogden
@wrapt.decorator
def timing(wrapped, instance, args, kwargs):
  def wrapper(*args, **kw):
    ts = time.time_ns()
    ts_n = time.thread_time_ns()
    ts_p = time.process_time_ns()
    result = wrapped(*args, **kw)
    te = time.time_ns()
    te_n = time.thread_time_ns()
    te_p = time.process_time_ns()
    # log.info(f"TIMING: {wrapped.__name__}({[str(i)[:100] for i in args], kw}): {(te-ts)/1000000.0 : 0.3f}ms {(te_n-ts_n)/1000000.0 : 0.3f}ms {(te_p-ts_p)/1000000.0 : 0.3f}ms")
    log.info(f"TIMING: {wrapped.__name__}: {(te-ts)/1000000.0 : 0.3f}ms {(te_n-ts_n)/1000000.0 : 0.3f}ms {(te_p-ts_p)/1000000.0 : 0.3f}ms")
    
    return result, ((te-ts)/1000000.0)
  return wrapper(*args, **kwargs)

@timing
def get_data(*args, **kwargs):
  # From https://github.com/tensorflow/datasets/blob/master/docs/keras_example.ipynb
  (ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
  )
  return ds_train, ds_test, ds_info

@timing
def process_data(ds_train, ds_test, ds_info, batch_size=128, *args, **kwargs):
  def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label
  
  if "normalize_input" in kwargs and kwargs["normalize_input"]:
    ds_train = ds_train.map(
      normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  ds_train = ds_train.cache()
  ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
  ds_train = ds_train.batch(batch_size)
  ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
  
  if "normalize_input" in kwargs and kwargs["normalize_input"]:
    ds_test = ds_test.map(
      normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  ds_test = ds_test.batch(batch_size)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
  
  return ds_train, ds_test

@timing
def get_model(*args, **kwargs):
  
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
  for _ in range(kwargs["num_dense_layers"]):
    model.add(tf.keras.layers.Dense(kwargs["num_dense_units"], activation='elu'))
  model.add(tf.keras.layers.Dense(10))
  
  model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy', 'sparse_categorical_crossentropy'])
  
  return model

@timing
def train_model(model, ds_train, *args, **kwargs) -> tf.keras.callbacks.History:
  if kwargs["gpu_train"]:
    with tf.device("/GPU:0"):
      history = model.fit(
        ds_train,
        epochs=(5 if "num_epochs" not in kwargs else kwargs["num_epochs"])
      )
  else:
    # with tf.device("/CPU:0"):
    history = model.fit(
      ds_train,
      epochs=(5 if "num_epochs" not in kwargs else kwargs["num_epochs"])
    )
  return history

@timing
def evaluate_model(model, ds_test, *args, **kwargs):
  if kwargs["gpu_test"]:
    with tf.device("/GPU:0"):
      model.evaluate(ds_test)
  else:
    with tf.device("/CPU:0"):
      model.evaluate(ds_test)


def run_test(num_epochs, normalize_input, *args, **kwargs):
  (ds_train, ds_test, ds_info), time__get_data = get_data(**kwargs)
  (ds_train, ds_test), time__process_data = process_data(
    ds_train,
    ds_test,
    ds_info,
    batch_size=1024,
    normalize_data=normalize_input,
    **kwargs
  )
  
  #creates the NNM
  (model), time__get_model = get_model(**kwargs)
  
  history, time__train_model = train_model(model, ds_train, num_epochs=num_epochs, **kwargs)
  _, time__evaluate_model = evaluate_model(model, ds_test, **kwargs)
  
  print(history.history.keys())
  
  return_dict = {
    "time__get_data" : time__get_data,
    "time__process_data" : time__process_data,
    "time__get_model" : time__get_model,
    "time__train_model" : time__train_model,
    "time__evaluate_model" : time__evaluate_model,
    #"model_accuracy" : model.
  }
  
  for key in history.history.keys():
    return_dict[key] = history.history[key]
  
  return return_dict

def write_to_csv(fid, results):
  for i, key in enumerate(results.keys()):
    if key == "validation_accuracies":
      continue
    fid.write(results[key])

  # I have no idea what the code under here is doing. Ask sam for clarification  
  with open(f"{results['val']}-{results[i].zfill(3)}.csv", "w") as run_specific_fid:
      # todo: add in column headers
      for epoch_num, epoch_accuracy in enumerate(results["validation_accuracies"]):
          run_specific_fid.write(f"{epoch_num},{epoch_accuracy}")

def add_in_hyperparameters(results, hyperparams):
  return results
def main():
  flags = parse_flags()
  with open("temp.txt", "w") as fid:
    for i in range(flags.num_trials):
      results = run_test(**vars(flags))
      # pprint(results)
      # todo: Write results out to CSV file
      results = add_in_hyperparameters(results, hyperparams={})
      #write_to_csv(fid, results)
      pprint(results)

# This makes the script launch the main function.
if __name__ == "__main__":
  main()

