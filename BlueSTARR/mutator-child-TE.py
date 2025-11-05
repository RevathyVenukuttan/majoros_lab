#!/usr/bin/env python
#========================================================================
# BlueSTARR Version 0.1
# Adapted from DeepSTARR by Bill Majoros (bmajoros@alumni.duke.edu)
#========================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gc
import gzip
import time
import math
import tensorflow as tf
tf.autograph.set_verbosity(1)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import keras
import keras.layers as kl
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers import BatchNormalization, InputLayer, Input, LSTM, GRU, Bidirectional, Add, Concatenate, LayerNormalization, MultiHeadAttention
import keras_nlp
from keras_nlp.layers import SinePositionEncoding, RotaryEmbedding, TransformerEncoder
from keras import models
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
import keras.backend as backend
from keras.backend import int_shape
import pandas as pd
import numpy as np
import sys
import random
from scipy import stats
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import ProgramName
from Rex import Rex
rex=Rex()



#========================================================================
#                                GLOBALS
#========================================================================
config=None
NUM_DNA=None # number of DNA replicates
NUM_RNA=None # number of RNA replicates
#RANDOM_SEED=1234
ALPHA={"A":0,"C":1,"G":2,"T":3}
BATCH_SIZE=1
tf.compat.v1.disable_eager_execution()
custom_objects = {
    "RotaryEmbedding":RotaryEmbedding,
    "TransformerEncoder":TransformerEncoder
    }

#=========================================================================
#                                main()
#=========================================================================
def main(infile,modelFilestem):
    #startTime=time.time()
    print(f"[INFO] Loading model from: {modelFilestem}", flush=True)
    print(f"[INFO] Reading input file: {infile}", flush=True)

    custom_objects = {
    "RotaryEmbedding":RotaryEmbedding,
    "TransformerEncoder":TransformerEncoder
    }

    # Load model
    model=None
    with open(modelFilestem+'.json', "r") as json_file:
        model_json=json_file.read()
    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
    model.load_weights(modelFilestem+'.h5') 

    print(f"[INFO] Model loaded successfully", flush=True)
    print(model.summary())

    # Load data
    IN=open(infile,"rt")
    recs=[]
    for line in IN:
        rec=line.rstrip().split()
        if(len(rec)<6): continue
        recs.append(rec)
    
    if not recs:
        print(f"[WARNING] No valid input records found.", flush=True)
        return

    print(f"[INFO] Number of sequences: {len(recs)}", flush=True)
    
    X=oneHot(recs)

    print(f"[INFO] Input shape to model: {X.shape}", flush=True)

    batchSize=8
    try:
        print(f"Running prediction on {len(recs)} sequences with batch size {batchSize}", flush=True)
        pred=model.predict(X,batch_size=batchSize,verbose=0)
        print(f"Predictions shape: {pred.shape}", flush=True)
    except Exception as e:
        print(f"Prediction failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    if len(pred)!=len(recs):
        print(f"[WARNING] Prediction count {len(pred)} doesn't match input count {len(recs)}", flush=True)

    
    numRecs=len(recs)
    for i in range(numRecs):
        # print(pred[i]); print(pred[i][0]); print(pred[i][0][0])
        # y=pred[i][0][0] #.numpy()
        rec=recs[i]
        (ID,actualInterval,pos,ref,allele,seq)=rec
        try:
            y=float(pred[i][0][0])
        except Exception as e:
            print(f"[ERROR] Could not extract prediction value at index {i}:{e}", file=sys.stderr, flush=True)
            continue
        print(ID,actualInterval,pos,ref,allele,y,sep="\t")
        
#========================================================================
#                               FUNCTIONS
#========================================================================
def oneHot(recs):
    firstRec=recs[0]
    (ID,actualInterval,pos,ref,allele,seq)=firstRec
    L=len(seq)
    numRecs=len(recs)
    X=np.zeros((numRecs,L,4))
    for j in range(numRecs):
        rec=recs[j]
        (ID,actualInterval,pos,ref,allele,seq)=rec
        for i in range(L):
            c=seq[i]
            cat=ALPHA.get(c,-1)
            if(cat>=0): X[j,i,cat]=1
    return X

#=========================================================================
#                         Command Line Interface
#=========================================================================
if(len(sys.argv)!=3):
    exit(ProgramName.get()+" <model-filestem> <data>\n")
(modelFilestem,infile)=sys.argv[1:]
main(infile,modelFilestem)


