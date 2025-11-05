#!/usr/bin/env python3
"""
Predict variant effects from a CRE list using a Keras model (<stem>.json + <stem>.h5).
"""
import argparse
import os
import sys
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import List, Tuple

import json, re, logging, importlib
import numpy as np
import tensorflow as tf

ALPHABET = {"A": 0, "C": 1, "G": 2, "T": 3}

############################## Data classes ##############################

@dataclass
class CrePosition:
    pos: int
    ref: str
    alleles: List[str]

@dataclass
class CRE:
    chrom: str
    begin: int
    end: int
    positions: List[CrePosition] = field(default_factory=list)

################################ Parsers #################################

def parse_cre_line(line: str) -> CRE:
    line = line.strip()
    if not line:
        raise ValueError("Empty line encountered in CRE file")
    fields = line.split()  # tabs or spaces
    region = fields[0]
    chrom, coords = region.split(":")
    beg, end = coords.split("-")
    cre = CRE(chrom=chrom, begin=int(beg), end=int(end))
    for field in fields[1:]:
        # Pattern: 12345:ref=G:G,C,A,T
        left, right = field.split(":ref=")
        pos = int(left)
        ref, alleles_csv = right.split(":")
        alleles = alleles_csv.split(",")
        cre.positions.append(CrePosition(pos=pos, ref=ref.upper(), alleles=[a.upper() for a in alleles]))
    return cre

def load_cres(path: str, max_n: int = -1) -> List[CRE]:
    cres: List[CRE] = []
    with open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cres.append(parse_cre_line(line))
            if max_n > 0 and len(cres) >= max_n:
                break
    if not cres:
        raise ValueError("No CREs parsed from input file")
    return cres

############################# Genome extraction ############################

def run_twoBitToFa(twobit_path: str, coords_path: str, out_fasta: str, twobittofa: str) -> None:
    cmd = [twobittofa, "-noMask", f"-seqList={coords_path}", twobit_path, out_fasta]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr.decode(errors="ignore"))
        raise

################################# FASTA ####################################

def read_fasta(path: str) -> List[Tuple[str, str]]:
    records = []
    with open(path, "rt") as f:
        header = None
        seq_chunks: List[str] = []
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            records.append((header, "".join(seq_chunks)))
    return records

######################### Universal model loader #############################

def _import_attr(modname: str, attr: str):
    try:
        mod = importlib.import_module(modname)
        return getattr(mod, attr, None)
    except Exception:
        return None

def _auto_custom_objects(model_json: str) -> dict:
    co: dict = {}
    try:
        from tensorflow.keras import activations as _act
        if "gelu" in model_json and hasattr(_act, "gelu"):
            co["gelu"] = _act.gelu
        if "swish" in model_json and hasattr(_act, "swish"):
            co["swish"] = _act.swish
    except Exception:
        pass
    try:
        import tensorflow_addons as tfa  # type: ignore
        if "mish" in model_json and hasattr(tfa.activations, "mish"):
            co["mish"] = tfa.activations.mish
        for name in ("InstanceNormalization", "GroupNormalization", "WeightNormalization"):
            if name in model_json and hasattr(tfa.layers, name):
                co[name] = getattr(tfa.layers, name)
    except Exception:
        pass
    if "keras_nlp" in model_json:
        for name in ("SinePositionEncoding","PositionEmbedding","TransformerEncoder","TransformerDecoder"):
            obj = _import_attr("keras_nlp.layers", name)
            if obj is not None and name in model_json:
                co[name] = obj
    class_names = set(re.findall(r'"class_name"\s*:\s*"([^"]+)"', model_json))
    for cls in class_names:
        for mod in ("tensorflow_addons.layers","keras_nlp.layers"):
            obj = _import_attr(mod, cls)
            if obj is not None:
                co[cls] = obj
    return co

def load_model_from_stem(model_stem: str):
    json_path = model_stem + ".json"
    weights_path = model_stem + ".h5"
    with open(json_path, "r") as jf:
        model_json = jf.read()
    try:
        model = tf.keras.models.model_from_json(model_json)
    except Exception as first_err:
        custom_objects = _auto_custom_objects(model_json)
        if not custom_objects:
            raise first_err
        logging.info("Retrying model_from_json() with custom_objects: %s",
                     ", ".join(sorted(custom_objects.keys())))
        model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
    model.load_weights(weights_path)
    return model

############################# One-hot + predict ############################

def one_hot_batch(seqs: List[str]) -> np.ndarray:
    if not seqs:
        raise ValueError("No sequences to encode")
    L = len(seqs[0])
    for s in seqs:
        if len(s) != L:
            raise ValueError("All sequences must be the same length")
    X = np.zeros((len(seqs), L, 4), dtype=np.float32)
    for j, s in enumerate(seqs):
        s = s.upper()
        for i, c in enumerate(s):
            k = ALPHABET.get(c)
            if k is None:
                raise ValueError(f"Invalid base {c!r} at position {i}")
            X[j, i, k] = 1.0
    return X

################################# Main ######################################

def main():
    arg = argparse.ArgumentParser(description="Predict variant effects (variant-centered windows) using a Keras model (stem.json + stem.h5)")
    arg.add_argument("--cre", required=True, help="CRE file (e.g., Kircher_region_chunk.txt)")
    arg.add_argument("--model-stem", required=True, help="Path stem to model (expects <stem>.json and <stem>.h5)")
    arg.add_argument("--two-bit", required=True, help="Reference genome in .2bit format (e.g., hg38.2bit)")
    arg.add_argument("--seq-len", type=int, required=True, help="Fixed window length to extract around *each variant*")
    arg.add_argument("--output", default="-", help="Output TSV (default: stdout)")
    arg.add_argument("--job-size", type=int, default=128, help="Batch size for model.predict")
    arg.add_argument("--twobittofa", default="twoBitToFa", help="Path to UCSC twoBitToFa (default: in $PATH)")
    args = arg.parse_args()

    # Load model and auto-match length
    model = load_model_from_stem(args.model_stem)
    expected_len = model.input_shape[1]
    seq_len = expected_len
    if args.seq_len != expected_len:
        print(f"[WARN] Overriding --seq-len {args.seq_len} -> {expected_len} to match model", file=sys.stderr)

    # Load CREs
    cres = load_cres(args.cre)

    # Build a list of variant-centered items (one window per variant)
    items = []
    half = seq_len // 2
    for cre in cres:
        for posrec in cre.positions:
            begin = posrec.pos - half
            if begin < 0:
                begin = 0
            end = begin + seq_len
            items.append({
                "chrom": cre.chrom,
                "cre_id": f"{cre.chrom}:{cre.begin}-{cre.end}",
                "begin": begin,
                "end": end,
                "posrec": posrec,  # includes pos, ref, alleles
            })

    # Make temporary files for coordinates and fasta
    with tempfile.TemporaryDirectory() as tmpd:
        coords_path = os.path.join(tmpd, "coords.txt")
        fasta_path = os.path.join(tmpd, "sequences.fa")

        # Write one coordinate per variant
        with open(coords_path, "wt") as fh:
            fh.write("\n".join(f"{it['chrom']}:{it['begin']}-{it['end']}" for it in items) + "\n")

        run_twoBitToFa(args.two_bit, coords_path, fasta_path, args.twobittofa)

        fasta_records = read_fasta(fasta_path)
        if len(fasta_records) != len(items):
            raise RuntimeError(f"twoBitToFa output count ({len(fasta_records)}) doesn't match variant count ({len(items)})")

        # Prepare output
        out = sys.stdout if args.output == "-" else open(args.output, "wt")
        try:
            print("ID\tactualInterval\tpos\tref\tallele\tprediction", file=out)  # keep header; remove if you prefer

            seqs_batch: List[str] = []
            meta_batch: List[Tuple[str, str, int, str, str]] = []  # ID, actual, pos, ref, allele

            for (header, seq), it in zip(fasta_records, items):
                seq = seq.upper()
                pos = it["posrec"].pos
                local = pos - it["begin"]
                if local < 0 or local >= len(seq):
                    raise ValueError(f"Position {pos} outside extracted window {it['chrom']}:{it['begin']}-{it['end']}")
                ref_base = seq[local]
                listed_ref = it["posrec"].ref
                if ref_base != listed_ref:
                    raise ValueError(
                        f"Ref mismatch at {it['chrom']}:{pos}: genome={ref_base} vs listed ref={listed_ref}"
                    )
                for allele in it["posrec"].alleles:
                    if allele not in ALPHABET:
                        continue
                    alt_seq = seq if allele == ref_base else (seq[:local] + allele + seq[local+1:])
                    seqs_batch.append(alt_seq)
                    meta_batch.append((it["cre_id"], f"{it['chrom']}:{it['begin']}-{it['end']}", pos, ref_base, allele))

                    if len(seqs_batch) >= args.job_size:
                        X = one_hot_batch(seqs_batch)
                        y = np.asarray(model.predict(X, batch_size=len(seqs_batch), verbose=0)).reshape((-1,))
                        for (ID, actual, p, r, a), yhat in zip(meta_batch, y):
                            print(f"{ID}\t{actual}\tpos={p}\tref={r}\t{a}\t{float(yhat)}", file=out)
                        seqs_batch.clear(); meta_batch.clear()

            if seqs_batch:
                X = one_hot_batch(seqs_batch)
                y = np.asarray(model.predict(X, batch_size=len(seqs_batch), verbose=0)).reshape((-1,))
                for (ID, actual, p, r, a), yhat in zip(meta_batch, y):
                    print(f"{ID}\t{actual}\tpos={p}\tref={r}\t{a}\t{float(yhat)}", file=out)
        finally:
            if out is not sys.stdout:
                out.close()

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")
    main()
