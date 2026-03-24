#!/usr/bin/env python
#========================================================================
# BlueSTARR-multitask Version 0.1
#
# Adapted from DeepSTARR by Bill Majoros (bmajoros@alumni.duke.edu)
# and Alexander Thomson.
# load trained model and do predictions for testing data 
# Modified by Revathy Venukuttan - 6 Feb 2026
#========================================================================

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import sys
import time
import gzip
import os

from keras.models import model_from_json
from NeuralConfig import NeuralConfig
from Rex import Rex
import SequenceHelper

rex = Rex()
config = None
NUM_DNA = None
NUM_RNA = None
EPS = 1e-10


# ============================================================
#                     DATA LOADING
# ============================================================

def loadFasta(fasta_path, stop_at=None):
    fastas = []
    seq = None
    header = None

    opener = gzip.open if fasta_path.endswith(".gz") else open
    with opener(fasta_path, "rt") as IN:
        for line in IN:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if seq is not None:
                    fastas.append([header, seq])
                    if stop_at and len(fastas) >= stop_at:
                        break
                header = line[1:]
                seq = ""
            else:
                seq += line.upper()

        if seq is not None and (not stop_at or len(fastas) < stop_at):
            fastas.append([header, seq])

    return pd.DataFrame({
        "idx": [e[0].split(" ")[0] for e in fastas],
        "sequence": [e[1] for e in fastas],
    })


def loadCounts(filename, maxCases):
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(filename, "rt") as IN:
        header = IN.readline().strip()

        if not rex.find(r"DNA=([,\d]+)\s+RNA=([,\d]+)", header):
            raise Exception(f"Cannot parse counts file header: {header}")

        DNAreps = [int(x) for x in rex[1].split(",")]
        RNAreps = [int(x) for x in rex[2].split(",")]

        lines = []
        for i, line in enumerate(IN):
            fields = [float(x) for x in line.rstrip().split()]
            lines.append(fields)
            if maxCases and i >= maxCases - 1:
                break

    Y = tf.cast(pd.DataFrame(lines), tf.float32)
    return DNAreps, RNAreps, Y


def prepare_test_input(subdir, config):
    fasta_file = os.path.join(subdir, "test.fasta.gz")
    counts_file = os.path.join(subdir, "test-counts.txt.gz")

    print("Loading FASTA...", flush=True)
    fasta = loadFasta(fasta_file, stop_at=config.MaxTest)

    if fasta.shape[0] == 0:
        raise Exception("No sequences found in test FASTA.")

    seqlen = len(fasta.sequence.iloc[0])

    print("One-hot encoding...", flush=True)
    seq_matrix = SequenceHelper.do_one_hot_encoding(
        fasta.sequence,
        seqlen,
        SequenceHelper.parse_alpha_to_seq
    )
    X = np.nan_to_num(seq_matrix).astype(np.float32)

    print("Loading counts...", flush=True)
    DNAreps, RNAreps, Y = loadCounts(counts_file, config.MaxTest)

    global NUM_DNA, NUM_RNA
    NUM_DNA = DNAreps
    NUM_RNA = RNAreps

    return X, Y, fasta.idx


# ============================================================
#                TRUE VALUES IN LOG SPACE
# ============================================================

def true_log_theta_for_task(Y_np, taskNum):
    """
    Returns true log(theta) for the given task.
    - If useCustomLoss=0, Y_np is already [N, numTasks] of log(theta) per task.
    - If useCustomLoss=1, Y_np is [N, sum(DNAreps+RNAreps)] of replicate counts,
      so compute naiveTheta = mean(RNA)/mean(DNA) and take log.
    """
    if not config.useCustomLoss:
        # Y is already log(theta) per task
        return Y_np[:, taskNum]

    # Y contains replicate counts across tasks: [DNA reps..., RNA reps...] per task
    a = 0
    for i in range(taskNum):
        a += NUM_DNA[i] + NUM_RNA[i]
    b = a + NUM_DNA[taskNum]
    c = b + NUM_RNA[taskNum]

    DNA = Y_np[:, a:b]
    RNA = Y_np[:, b:c]

    avgX = np.mean(DNA, axis=1)
    avgY = np.mean(RNA, axis=1)

    naiveTheta = (avgY + EPS) / (avgX + EPS)
    return np.log(naiveTheta)


# ============================================================
#                         MAIN
# ============================================================

def main(configFile, subdir, modelFilestem):
    start = time.time()

    global config
    config = NeuralConfig(configFile)

    # Load test data
    X_test, Y_test, idx = prepare_test_input(subdir, config)
    Y_np = Y_test.numpy()

    # Load model architecture
    print("Loading model architecture (.json)...", flush=True)
    with open(modelFilestem + ".json", "r") as f:
        model_json = f.read()

    model = model_from_json(model_json)

    # Load weights
    print("Loading weights (.h5)...", flush=True)
    model.load_weights(modelFilestem + ".h5")

    # Predict
    print("Predicting...", flush=True)
    preds = model.predict(X_test, batch_size=config.BatchSize)

    # Handle single-task output
    if not isinstance(preds, list):
        preds = [preds]

    # Write outputs per task
    for taskNum, taskName in enumerate(config.Tasks):
        pred_vals = preds[taskNum].squeeze()
        true_vals = true_log_theta_for_task(Y_np, taskNum)

        mse = np.mean((true_vals - pred_vals) ** 2)

        df = pd.DataFrame({
            "idx": idx,
            "true": true_vals,
            "predicted": pred_vals
        })

        outfile = f"{modelFilestem}_{taskName}_predictions.txt"
        df.to_csv(outfile, sep="\t", index=False)

        print(f"{taskName} MSE (log space) = {mse:.6f}", flush=True)
        print(f"Saved: {outfile}", flush=True)

    print("Done.")
    print("Elapsed time:", round((time.time() - start) / 60, 2), "minutes")


# ============================================================
#                    Command Line Interface
# ============================================================

if len(sys.argv) != 4:
    exit("Usage: BlueSTARR-predict.py <config> <data-subdir> <model-filestem>")

configFile, subdir, modelFilestem = sys.argv[1:]
main(configFile, subdir, modelFilestem)
