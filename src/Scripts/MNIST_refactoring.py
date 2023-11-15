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

# Sam recommended using a namespace instead of a globals. args should already be a namespace

#This is the logging algorithm given to me from Dr. Ogden
@wrapt.decorator
def timing(wrapped, instance, args, kwargs):
    def wrapper(*args, **kw):
        # ts = time.time_ns()
        # ts_n = time.thread_time_ns()
        # ts_p = time.process_time_ns()
        result = wrapped(*args, **kw)
        # te = time.time_ns()
        # te_n = time.thread_time_ns()
        # te_p = time.process_time_ns()
        # log.info(f"TIMING: {wrapped.__name__}({[str(i)[:100] for i in args], kw}): {(te-ts)/1000000.0 : 0.3f}ms {(te_n-ts_n)/1000000.0 : 0.3f}ms {(te_p-ts_p)/1000000.0 : 0.3f}ms")
        
        return result
    return wrapper(*args, **kwargs)

# Creates a dataset using a tensorsflow dataset (tfds)
# ds = tfds.load('mnist', split='train').batch(32).prefetch()
def main():

    #Download dataset
    #make model
    #train model

    #can use a premade model


    return

    flags = parse_flags()
    time_start = time.time()
    for _ in range(10):
        # This is all to load the MNIST data
        ml = MNIST(32, 10)
        ml.load_data()
        ds = ml.ds
        ds_info = ml.ds_info
        print(f"ds: {ds}")
	    
        # This is to train a model. As more models are added, more paser flags should be added.
        input_shape = ds_info.features['image'].shape
        num_classes = ds_info.features['label'].num_classes

        #Create the NN by passing in the inputshape and the number of classes.
        nn_model = SimpleNNModel(input_shape, num_classes)

        # Compile and train the model
        nn_model.compile_model()
        nn_model.train_model(ds, epochs=5)

        # Evaluate the model
        result = nn_model.evaluate_model(ds)
        print("Evaluation Result:", result)

        
    log.info(f"Overall: {(time.time() - time_start) / 10.0}")

def split_data(ds):
    pass

def parse_flags():
    parser = argparse.ArgumentParser()#description='Determine where to use CPU in this machine learning algorithm.')
    # action='store_true' results in just calling -a to give a true value
    parser.add_argument('-a', "--all", action='store_true', dest="all_on_cpu", help = 'Entire algorithm runs on the CPU')
    args = parser.parse_args()
    # 
    if args.all_on_cpu:
        # I will need an if statment with "with tf.device('/CPU:0'):" and without that depending on whether I want it to run on CPU or not
        args.train_on_cpu = True
        print("Entire MLA will run on CPU")
    else:
        print("Entire MLA will run on GPU")
    return args


class MNIST:
	
    @timing    
    def __init__(self, batch_size, buffer_amount, prefetch=False):
        """
        Initalize an MNIST data data

        Args:
            batch_size (int): download data to batches for training - passed through to downloading
            buffer_amount (int): used in prefetch to set download buffer size - passed through to preping the data
            prefetch (bool) : ...

        Returns:
            None            
        """
        self.batch_size = batch_size
        self.buffer_amount = buffer_amount
        self.ds = None
        self.prefetch = prefetch

    @timing
    def load_data(self):
        """
        Loads in data to train our model

        Ags:
            None

        Returns:
            None
        """
        (self.ds, self.ds_info) = tfds.load('mnist', split='train', with_info=True)
        print(f"self.ds: {self.ds}")
        if False:
            if self.prefetch:
                self.ds = self.ds.batch(self.batch_size).prefetch(buffer_size = self.buffer_amount)
            else:
                self.ds = self.ds.batch(self.batch_size)
        #return self.ds

    @staticmethod
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label




class SimpleNNModel:
    """
    This is here to hopefully make thing easier to load different models depending on our needs.
    """
    def __init__(self, input_shape, num_of_classes):
        self.input_shape = input_shape
        self.num_of_classes = num_of_classes
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        # model.add(layers.Flatten())
        model.add(layers.Dense(128, activation = 'relu'))
        model.add(layers.Dense(64, activation = 'relu'))
        model.add(layers.Dense(self.num_of_classes, activation ='softmax'))
        return model
    
    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self, dataset, epochs=5):
        train, test = tf.keras.datasets.mnist.load_data()
        # print(f"train: {train}")
        # print(f"dataset: {dataset}")
        x = dataset.map(lambda i: i['image'])
        y = dataset.map(lambda i: i['label'])

        # print(f"dataset[0] : {list(dataset)[0]}")
        self.model.fit(train, epochs=epochs)

    def evaluate_model(self, dataset):
        return self.model.evaluate(dataset)

    
# This makes the script launch the main function.
if __name__ == "__main__":
	main()

