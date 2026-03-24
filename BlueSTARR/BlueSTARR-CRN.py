#!/usr/bin/env python
#========================================================================
# BlueSTARR-multitask Version 0.2 (Causal + Residual, STARR-first-pass)
#
# First-pass biology-consistent hierarchy for STARR-seq:
#   DNA -> TF-latent (motif/grammar proxy) -> Enhancer activity
# PLUS a residual (unconstrained) skip:
#   DNA -> Enhancer activity
#
# Trained on STARR-seq (e.g., K562 wgSTARR) to predict log(theta)
# where theta ~ RNA/DNA (your current convention).
#
# Notes:
# - Intermediate "TF-latent" is a learned latent representation, NOT literal TF binding
#   unless you later supervise it with ChIP/ATAC.
# - Keeps your existing data loading, loss closures, CLI, and output naming behavior.
# - Supports multi-task in the same way as before (one output per config.Tasks entry),
#   but for K562 wgSTARR you typically want Tasks=["K562"].
#========================================================================

import gzip
import time
import tensorflow as tf
import keras
import keras.layers as kl
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Dropout, Dense, Activation, Flatten
from keras.layers import BatchNormalization, Input, Add, Concatenate, LayerNormalization, MultiHeadAttention
import keras_nlp
from keras import models
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History
import keras.backend as backend
from keras.backend import int_shape

import pandas as pd
import numpy as np
import ProgramName
import sys
import IOHelper
import SequenceHelper
import random
from scipy import stats
from scipy.stats import spearmanr
from NeuralConfig import NeuralConfig
from Rex import Rex
rex = Rex()

#========================================================================
#                                GLOBALS
#========================================================================
config = None
NUM_DNA = None  # array: numbers of DNA replicates in each task
NUM_RNA = None  # array: numbers of RNA replicates in each task
EPSILON = tf.cast(1e-10, tf.float32)

#========================================================================
#                          Helper: Trainable gate mix
#========================================================================
class GateMix(kl.Layer):
    """
    y = sigmoid(g) * y_causal + (1-sigmoid(g)) * y_resid
    g is a learned scalar per task/head.
    """
    def __init__(self, name=None, init_logit=0.0):
        super().__init__(name=name)
        self.init_logit = init_logit

    def build(self, input_shape):
        self.logit = self.add_weight(
            name="gate_logit",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init_logit),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        y_causal, y_resid = inputs
        g = tf.sigmoid(self.logit)
        return g * y_causal + (1.0 - g) * y_resid

#=========================================================================
#                                main()
#=========================================================================
def main(configFile, subdir, modelFilestem):
    startTime = time.time()

    # Load hyperparameters from configuration file
    global config
    config = NeuralConfig(configFile)

    # Load data
    print("loading data", flush=True)
    shouldRevComp = (config.RevComp == 1)

    (X_train_sequence, X_train_seq_matrix, X_train, Y_train, idx_train) = \
        prepare_input("train", subdir, shouldRevComp, config.MaxTrain, config)
    (X_valid_sequence, X_valid_seq_matrix, X_valid, Y_valid, idx_val) = \
        prepare_input("validation", subdir, shouldRevComp, config.MaxTrain, config)
    (X_test_sequence, X_test_seq_matrix, X_test, Y_test, idx_test) = \
        prepare_input("test", subdir, shouldRevComp, config.MaxTest, config) \
        if (config.ShouldTest != 0) else (None, None, None, None, None)

    seqlen = X_train.shape[1]

    # Build model
    model = BuildModel(seqlen)
    model.summary()

    # Train
    history = None
    if config.Epochs > 0:
        print("Training...", flush=True)
        print("Training set:", X_train.shape)
        (model, history) = train(model, X_train, Y_train, X_valid, Y_valid)
        print("Done training", flush=True)
        print("loss", history.history.get('loss', None))
        print("val_loss", history.history.get('val_loss', None))

    # Save model
    model_json = model.to_json()
    with open(modelFilestem + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(modelFilestem + ".h5")

    # Test and report accuracy
    if config.ShouldTest != 0:
        numTasks = len(config.Tasks)
        for i in range(numTasks):
            summary_statistics(X_test, Y_test, "Test", i, numTasks,
                               config.Tasks[i], model, idx_test, modelFilestem)

    if history is not None and "val_loss" in history.history:
        print('Min validation loss:', round(min(history.history['val_loss']), 4))

    # Report elapsed time
    endTime = time.time()
    minutes = (endTime - startTime) / 60
    print("Elapsed time:", round(minutes, 2), "minutes")

#========================================================================
#                           Evaluation helpers
#========================================================================
def summary_statistics(X, Y, setName, taskNum, numTasks, taskName, model, idx, modelFilestem):
    pred = model.predict(X, batch_size=config.BatchSize)

    if config.useCustomLoss:
        naiveTheta, cor = naiveCorrelation(Y, pred, taskNum, numTasks)
        df = pd.DataFrame({'idx': idx, 'true': tf.math.log(naiveTheta), 'predicted': pred[taskNum].squeeze() if numTasks > 1 else pred.squeeze()})
        mse = np.mean((df['true'] - df['predicted']) ** 2)
    else:
        y_true = Y[:, taskNum] if (len(Y.shape) == 2 and Y.shape[1] > 1) else tf.reshape(Y, [-1])
        y_true = y_true.numpy().ravel()

        y_final  = pred[0].squeeze()
        y_causal = pred[1].squeeze()
        y_resid  = pred[2].squeeze()

        cor = stats.spearmanr(tf.math.exp(y_final), tf.math.exp(y_true))
        df = pd.DataFrame({'idx': idx, 'true': y_true, 'pred_final': y_final, 'pred_causal': y_causal, 'pred_residual': y_resid})
        mse = np.mean((df['true'] - df['pred_final']) ** 2)

    out_tsv = f"{modelFilestem}.{taskName}.txt" if len(config.Tasks) > 1 else f"{modelFilestem}.txt"
    df.to_csv(out_tsv, index=False, sep='\t')

    print(taskName + " rho=", cor.statistic, "p=", cor.pvalue)
    print(taskName + " mse=", mse)

def naiveCorrelation(y_true, y_pred, taskNum, numTasks):
    a = 0
    for i in range(taskNum):
        a += NUM_DNA[i] + NUM_RNA[i]
    b = a + NUM_DNA[taskNum]
    c = b + NUM_RNA[taskNum]
    DNA = y_true[:, a:b]
    RNA = y_true[:, b:c]
    avgX = tf.reduce_mean(DNA, axis=1)
    avgY = tf.reduce_mean(RNA, axis=1)
    naiveTheta = avgY / avgX

    if numTasks == 1:
        cor = stats.spearmanr(tf.math.exp(y_pred.squeeze()), naiveTheta)
    else:
        cor = stats.spearmanr(tf.math.exp(y_pred[taskNum].squeeze()), naiveTheta)
    return naiveTheta, cor

#========================================================================
#                               LOSSES
#========================================================================
def log(x): return tf.math.log(x)
def logGam(x): return tf.math.lgamma(x)

def logLik(sumX, numX, Yj, logTheta, alpha, beta, numRNA):
    n = tf.shape(sumX)[0]
    sumX = tf.tile(tf.reshape(sumX, [n, 1]), [1, numRNA])
    theta = tf.math.exp(logTheta)  # model predicts log(theta)
    LL = (sumX + alpha) * log(beta + numX) + logGam(Yj + sumX + alpha) + Yj * log(theta) \
         - logGam(sumX + alpha) - logGam(Yj + 1) - (Yj + sumX + alpha) * log(theta + beta + numX)
    return tf.reduce_sum(LL, axis=1)

@tf.autograph.experimental.do_not_convert
def makeClosure(taskNum):
    a = 0
    for i in range(taskNum):
        a += NUM_DNA[i] + NUM_RNA[i]
    b = a + NUM_DNA[taskNum]
    c = b + NUM_RNA[taskNum]

    @tf.autograph.experimental.do_not_convert
    def loss(y_true, y_pred):
        global EPSILON
        DNA = y_true[:, a:b]
        RNA = y_true[:, b:c]
        sumX = tf.reduce_sum(DNA, axis=1)
        LL = -logLik(sumX, b - a, RNA, y_pred, EPSILON, EPSILON, NUM_RNA[taskNum])
        return LL

    return loss

#========================================================================
#                          FASTA / COUNTS LOADING
#========================================================================
def generate_complementary_sequence(sequence):
    comp = []
    for b in sequence:
        if b == "A": comp.append("T")
        elif b == "T": comp.append("A")
        elif b == "C": comp.append("G")
        elif b == "G": comp.append("C")
        elif b == "N": comp.append("N")
        else: raise ValueError(f"Cannot convert base {b} to complement base!")
    return ''.join(comp)

def loadFasta(fasta_path, as_dict=False, uppercase=False, stop_at=None, revcomp=False):
    fastas = []
    seq = None
    header = None
    for r in (gzip.open(fasta_path) if fasta_path.endswith(".gz") else open(fasta_path)):
        if type(r) is bytes: r = r.decode("utf-8")
        r = r.strip()
        if r.startswith(">"):
            if seq is not None and header is not None:
                fastas.append([header, seq])
                if stop_at is not None and len(fastas) >= stop_at:
                    break
            seq = ""
            header = r[1:]
        else:
            if seq is not None:
                seq += r.upper() if uppercase else r
            else:
                seq = r.upper() if uppercase else r

    if stop_at is not None and len(fastas) < stop_at:
        fastas.append([header, seq])
    elif stop_at is None:
        fastas.append([header, seq])

    if as_dict:
        return {h: s for h, s in fastas}

    if revcomp:
        for rec in fastas:
            rc = generate_complementary_sequence(rec[1])
            rec[1] = rec[1] + "NNNNNNNNNNNNNNNNNNNN" + rc

    return pd.DataFrame({
        'location': [e[0] for e in fastas],
        'idx': [e[0].split(' ')[0] for e in fastas],
        'sequence': [e[1] for e in fastas]
    })

def loadCounts(filename, maxCases, config):
    IN = gzip.open(filename) if filename.endswith(".gz") else open(filename)
    header = IN.readline()
    if type(header) is bytes: header = header.decode("utf-8")
    if not rex.find("DNA=([,\\d]+)\\s+RNA=([,\\d]+)", header):
        raise Exception("Can't parse counts file header: " + header)

    DNAreps = [int(x) for x in rex[1].split(",")]
    RNAreps = [int(x) for x in rex[2].split(",")]
    linesRead = 0
    lines = []
    for line in IN:
        if type(line) is bytes: line = line.decode("utf-8")
        fields = line.rstrip().split()
        fields = [float(x) for x in fields]  # normalized data
        if config.useCustomLoss:
            lines.append(fields)
        else:
            lines.append(computeNaiveTheta(fields, DNAreps, RNAreps))
        linesRead += 1
        if linesRead >= maxCases:
            break
    return (DNAreps, RNAreps, np.array(lines))

def computeNaiveTheta(line, DNAreps, RNAreps):
    numTasks = len(DNAreps)
    a = 0
    rec = []
    for i in range(numTasks):
        b = a + DNAreps[i]
        c = b + RNAreps[i]
        DNA = line[a:b]
        RNA = line[b:c]
        avgX = sum(DNA) / DNAreps[i]
        avgY = sum(RNA) / RNAreps[i]
        naiveTheta = avgY / avgX
        rec.append(tf.math.log(naiveTheta))  # log-scale
        a = c
    return rec

def prepare_input(setName, subdir, shouldRevComp, maxCases, config):
    file_seq = str(subdir + "/" + setName + ".fasta.gz")
    input_fasta = loadFasta(file_seq, uppercase=True, revcomp=shouldRevComp, stop_at=maxCases)
    sequence_length = len(input_fasta.sequence.iloc[0])
    print(sequence_length)

    seq_matrix = SequenceHelper.do_one_hot_encoding(
        input_fasta.sequence, sequence_length, SequenceHelper.parse_alpha_to_seq
    )
    X = np.nan_to_num(seq_matrix)
    X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    (DNAreps, RNAreps, Y) = loadCounts(subdir + "/" + setName + "-counts.txt.gz", maxCases, config)

    global NUM_DNA, NUM_RNA
    NUM_DNA = DNAreps
    NUM_RNA = RNAreps

    matrix = tf.cast(pd.DataFrame(Y), tf.float32)
    return (input_fasta.sequence, seq_matrix, X_reshaped, matrix, input_fasta.idx)

#========================================================================
#                          MODEL: causal + residual
#========================================================================
def BuildModel(seqlen):
    """
    Keeps your trunk (conv/attn/pooling/dense) and replaces the head with:

      Residual (unconstrained): shared -> enh_resid -> y_enh_resid
      Causal (STARR-consistent): shared -> tf_latent -> enh_causal -> y_enh_causal
      Combined: y_enh = gate * y_enh_causal + (1-gate) * y_enh_resid

    Outputs remain one per config.Tasks entry (names match task names),
    so your training + evaluation code doesn't need to change.
    """
    # ----------------------------
    # Input + shared trunk (unchanged structure)
    # ----------------------------
    inputLayer = Input(shape=(seqlen, 4), name="dna_onehot")
    x = inputLayer
    curr_len = float(seqlen)

    # Conv stack
    for i in range(config.NumConv):
        if config.KernelSizes[i] >= curr_len:
            continue
        dilation = 1 if i == 0 else config.DilationFactor
        if i > 0 and config.ConvDropout != 0:
            x = Dropout(config.DropoutRate, name=f"conv_dropout_{i}")(x)

        x = Conv1D(
            config.NumKernels[i],
            kernel_size=config.KernelSizes[i],
            padding=config.ConvPad,
            dilation_rate=dilation,
            name=f"conv_{i}",
        )(x)
        x = BatchNormalization(name=f"conv_bn_{i}")(x)
        x = Activation('relu', name=f"conv_relu_{i}")(x)

        if config.ConvPoolSize > 1 and curr_len > config.ConvPoolSize:
            x = MaxPooling1D(config.ConvPoolSize, name=f"conv_pool_{i}")(x)
            curr_len /= config.ConvPoolSize

    # Attention stack
    if config.NumAttn > 0:
        x = x + keras_nlp.layers.SinePositionEncoding(name="posenc")(x)

    for i in range(config.NumAttn):
        skip = x
        x = LayerNormalization(name=f"attn_ln_{i}")(x)
        x = MultiHeadAttention(
            num_heads=config.AttnHeads[i],
            key_dim=config.AttnKeyDim[i],
            name=f"attn_mha_{i}"
        )(x, x)
        x = Dropout(config.DropoutRate, name=f"attn_dropout_{i}")(x)
        if config.AttnResidualSkip != 0:
            x = Add(name=f"attn_resid_{i}")([x, skip])

    # Global pooling
    if config.GlobalMaxPool != 0:
        x = MaxPooling1D(int_shape(x)[1], name="global_maxpool")(x)
    if config.GlobalAvePool != 0:
        x = AveragePooling1D(int_shape(x)[1], name="global_avgpool")(x)

    # Flatten
    if config.Flatten != 0:
        x = Flatten(name="flatten")(x)

    # Dense stack
    if config.NumDense > 0:
        x = Dropout(config.DropoutRate, name="pre_dense_dropout")(x)

    for i in range(config.NumDense):
        x = Dense(config.DenseSizes[i], name=f"dense_{i}")(x)
        x = BatchNormalization(name=f"dense_bn_{i}")(x)
        x = Activation('relu', name=f"dense_relu_{i}")(x)
        x = Dropout(config.DropoutRate, name=f"dense_dropout_{i}")(x)

    shared = x  # [B, D]

    # ----------------------------
    # Head builder (small MLP + latent + pred)
    # ----------------------------
    # Optional config knobs; if not present, defaults used.
    LATENT = getattr(config, "CausalLatentDim", 128)
    HIDDEN = getattr(config, "CausalHiddenDim", 256)

    def head_block(inp, prefix):
        h = Dense(HIDDEN, activation="relu", name=f"{prefix}_h")(inp)
        h = Dropout(config.DropoutRate, name=f"{prefix}_drop")(h)
        z = Dense(LATENT, activation="relu", name=f"{prefix}_latent")(h)
        y = Dense(1, activation="linear", name=f"{prefix}_pred")(z)
        return z, y

    # ----------------------------
    # Per-task outputs (names match config.Tasks)
    # ----------------------------
    tasks = config.Tasks
    outputs = []
    losses = []
    loss_weights = []
    weights = [float(x) for x in config.TaskWeights]
    numTasks = len(tasks)

    for i in range(numTasks):
        task = tasks[i]

        # Residual enhancer
        _, y_enh_resid = head_block(shared, f"{task}_enh_resid")

        # "TF-latent" (motif/grammar proxy) then causal enhancer
        z_tf, _ = head_block(shared, f"{task}_tf_causal")
        _, y_enh_causal = head_block(Concatenate(name=f"{task}_enh_causal_in")([shared, z_tf]),
                                     f"{task}_enh_causal")

        # Combine with a learned gate
        y_enh = GateMix(name=f"{task}_enh_gated", init_logit=0.0)([y_enh_causal, y_enh_resid])

        # Output name must match the original task name for compatibility
        y_causal_out = kl.Lambda(lambda t: t, name=f"{task}_enh_causal_out")(y_enh_causal)
        y_resid_out  = kl.Lambda(lambda t: t, name=f"{task}_enh_resid_out")(y_enh_resid)
        y_final_out = kl.Lambda(lambda t: t, name=task)(y_enh)

        outputs.extend([y_final_out, y_causal_out, y_resid_out])

        main_loss = makeClosure(i) if config.useCustomLoss else "mse"
        
        losses.extend([main_loss, "mse", "mse"])
        loss_weights.extend([1.0, 0.0, 0.0])

    model = models.Model([inputLayer], outputs, name="BlueSTARR_CausalResidual_STARR")
    model.compile(
        Adam(learning_rate=config.LearningRate),
        run_eagerly=True,
        loss=losses,
        loss_weights=loss_weights
    )
    return model

#========================================================================
#                               TRAIN
#========================================================================
def train(model, X_train, Y_train, X_valid, Y_valid):
    earlyStop = EarlyStopping(
        patience=config.EarlyStop,
        monitor="val_loss",
        restore_best_weights=True
    )
    history = model.fit(
        X_train, Y_train,
        verbose=config.Verbose,
        validation_data=(X_valid, Y_valid),
        batch_size=config.BatchSize,
        epochs=config.Epochs,
        callbacks=[earlyStop, History()]
    )
    return (model, history)

#=========================================================================
#                         Command Line Interface
#=========================================================================
if __name__ == "__main__":
    if len(sys.argv) != 4:
        exit(ProgramName.get() + " <parms.config> <data-subdir> <out:model-filestem>\n")
    (configFile, subdir, modelFilestem) = sys.argv[1:]
    main(configFile, subdir, modelFilestem)
