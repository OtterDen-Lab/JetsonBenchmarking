##THIS WAS IMPORTED FROM ADV. DATA SCIENCE
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

def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize_input", action="store_true")
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
        log.info(f"TIMING: {wrapped.__name__}({[str(i)[:100] for i in args], kw}): {(te-ts)/1000000.0 : 0.3f}ms {(te_n-ts_n)/1000000.0 : 0.3f}ms {(te_p-ts_p)/1000000.0 : 0.3f}ms")
        
        return result
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
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

@timing
def train_model(model, ds_train, *args, **kwargs):
    model.fit(ds_train, epochs=5)

@timing
def evaluate_model(model, ds_test, *args, **kwargs):
    model.evaluate(ds_test)


def main():
    
    flags = parse_flags()
    
    ds_train, ds_test, ds_info = get_data()
    ds_train, ds_test = process_data(
        ds_train,
        ds_test,
        ds_info,
        batch_size=1024,
        normalize_data=flags.normalize_input
    )
    
    #creates the NNM
    model = get_model()
    
    train_model(model, ds_train)
    evaluate_model(model, ds_test)
    


# This makes the script launch the main function.
if __name__ == "__main__":
	main()

