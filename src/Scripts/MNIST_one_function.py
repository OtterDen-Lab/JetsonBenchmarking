# Non-machine learning classes
import time
import wrapt
import logging
import argparse

# Machine learning classes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import callbacks


#Allows for logging to happen.
logging.basicConfig()
log = logging.getLogger("common")
log.setLevel(logging.INFO)

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
def main():

    #----------------------------------------------#
    #This was the parse arguements function
    parser = argparse.ArgumentParser()#description='Determine where to use CPU in this machine learning algorithm.')
    # action='store_true' results in just calling -a to give a true value
    parser.add_argument('-a', "--all", action='store_true', dest="all_on_cpu", help = 'Entire algorithm runs on the CPU')
    args = parser.parse_args()

    if args.all_on_cpu:
        # I will need an if statment with "with tf.device('/CPU:0'):" and without that depending on whether I want it to run on CPU or not
        args.train_on_cpu = True
        print("Entire MLA will run on CPU")
    else:
        print("Entire MLA will run on GPU")
    #----------------------------------------------#


    #----------------------------------------------#
    #This was the init of the MNIST class. The arguements and defaults have been added to the variables.
    batch_size = 32
    buffer_amount = 10
    ds = None
    # prefetch = False
    #----------------------------------------------#

    #----------------------------------------------#
    #This was the load fuction of the MNIST class. The vars from the init are used here. I kept everything the same to keep prostarity. 
    (ds, ds_info) = tfds.load('mnist', split='train', with_info=True)
    print(f"ds: {ds}")
    # if False:
    #     if prefetch:
    #         ds = ds.batch(batch_size).prefetch(buffer_size = buffer_amount)
    #     else:
    #         ds = ds.batch(batch_size)
    print(f"DS shape: {ds}")
    #----------------------------------------------#

    #----------------------------------------------#
    #This was the normalize image function. This was not used so I commented it out.
    #tf.cast(image, tf.float32) / 255., label
    #----------------------------------------------#


    #This is the simple NN fucntions
    #----------------------------------------------#
    
    input_shape = ds_info.features['image'].shape
    num_of_classes = ds_info.features['label'].num_classes
    #model = self.build_model()
    #----------------------------------------------#

    #----------------------------------------------#
    #creates the NNM
    model = models.Sequential()
        # model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(num_of_classes, activation ='softmax'))
    #----------------------------------------------#

    #----------------------------------------------#
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    #----------------------------------------------#

    #----------------------------------------------#
    
    train, test = tf.keras.datasets.mnist.load_data()
        # print(f"train: {train}")
        # print(f"dataset: {dataset}")
    x = ds.map(lambda i: i['image'])
    y = ds.map(lambda i: i['label'])

        # print(f"dataset[0] : {list(dataset)[0]}")
    model.fit(train, epochs=5)
    #----------------------------------------------#

    #----------------------------------------------#
    model.evaluate(ds)
    #----------------------------------------------#



# This makes the script launch the main function.
if __name__ == "__main__":
	main()

