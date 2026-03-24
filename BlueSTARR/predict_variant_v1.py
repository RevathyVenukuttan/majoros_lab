#!/usr/bin/env python3

import argparse
import os
import sys
import gzip
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

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

@dataclass
class BirdVariant:
    variant_id: str   # e.g. "chr1@169611479"
    chrom: str
    pos: int          # 1-based
    ref: str
    alt: str

################################ Parsers #################################

def parse_cre_line(line):
    line = line.strip()
    if not line:
        raise ValueError("Empty line encountered in CRE file")
    fields = line.split()
    region = fields[0]
    chrom, coords = region.split(":")
    beg, end = coords.split("-")
    cre = CRE(chrom=chrom, begin=int(beg), end=int(end))
    for field in fields[1:]:
        left, right = field.split(":ref=")
        pos = int(left)
        ref, alleles_csv = right.split(":")
        alleles = alleles_csv.split(",")
        cre.positions.append(
            CrePosition(
                pos=pos,
                ref=ref.upper(),
                alleles=[a.upper() for a in alleles]
            )
        )
    return cre

def load_cres(path,max_n):
    cres = []
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

def parse_bird_line(line):
    """
    Expected BIRD input format:
      chr1@169611479    T   C
    """
    line = line.strip()
    if not line or line.startswith("#"):
        raise ValueError("Empty/comment line in BIRD file")
    fields = line.split()
    if len(fields) < 3:
        raise ValueError(f"BIRD line has <3 fields (need VariantID, ref, alt): {line}")

    variant_id, ref, alt = fields[:3]
    if "@" not in variant_id:
        raise ValueError(f"VariantID must be 'chr@pos': {variant_id}")
    chrom, pos_str = variant_id.split("@")
    return BirdVariant(
        variant_id=variant_id,
        chrom=chrom,
        pos=int(pos_str),
        ref=ref.upper(),
        alt=alt.upper(),
    )

def load_bird_variants(path,max_n):
    out = []
    with open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ("VariantID" in s) and ("p-reg" in s or "p_reg" in s):
                continue
            out.append(parse_bird_line(s))
            if max_n > 0 and len(out) >= max_n:
                break
    if not out:
        raise ValueError("No variants parsed from BIRD file")
    return out

############################# Genome extraction ############################

def run_twoBitToFa(twobit_path, coords_path, out_fasta, twobittofa):
    cmd = [twobittofa, "-noMask", f"-seqList={coords_path}", twobit_path, out_fasta]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr.decode(errors="ignore"))
        raise

################################# FASTA ####################################

def read_fasta(path):
    records = []

    if path.endswith(".gz"):
        fh = gzip.open(path, "rt")
    else:
        fh = open(path, "rt")

    with fh:
        header: Optional[str] = None
        seq_chunks: List[str] = []
        for line in fh:
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

def parse_interval_header(header):
    tokens = header.split()
    for tok in tokens:
        if "coord=" in tok and ":" in tok and "-" in tok:
            interval = tok.split("coord=")[-1]
            interval = interval.lstrip("/")
            chrom, coords = interval.split(":")
            beg_str, end_str = coords.split("-")
            return chrom, int(beg_str), int(end_str)

    for tok in tokens:
        if ":" in tok and "-" in tok:
            if "=" in tok:
                tok = tok.split("=")[-1]
            chrom, coords = tok.split(":")
            beg_str, end_str = coords.split("-")
            return chrom, int(beg_str), int(end_str)

    raise ValueError(
        f"Header must contain an interval like 'chr:start-end' or 'coord=chr:start-end', got: {header!r}"
    )

######################### Universal model loader #############################

def _import_attr(modname: str, attr: str):
    try:
        mod = importlib.import_module(modname)
        return getattr(mod, attr, None)
    except Exception:
        return None

def _auto_custom_objects(model_json):
    co = {}
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
        for name in ("SinePositionEncoding", "PositionEmbedding", "TransformerEncoder", "TransformerDecoder"):
            obj = _import_attr("keras_nlp.layers", name)
            if obj is not None and name in model_json:
                co[name] = obj
    class_names = set(re.findall(r'"class_name"\s*:\s*"([^"]+)"', model_json))
    for cls in class_names:
        for mod in ("tensorflow_addons.layers", "keras_nlp.layers"):
            obj = _import_attr(mod, cls)
            if obj is not None:
                co[cls] = obj
    return co

def load_model_from_stem(model_stem):
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

def one_hot_batch(seqs):
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
    parser = argparse.ArgumentParser(description="Predict sequence scores using a Keras model (stem.json + stem.h5)")
    parser.add_argument("--input", required=True, help="Input file: CRE lines, BIRD table, or FASTA")
    parser.add_argument("--input-format", choices=["legacy", "bird", "fasta"], default="legacy",
                        help="Input format: 'legacy', 'bird', or 'fasta'")
    parser.add_argument("--model-stem", required=True, help="Path stem to model (expects <stem>.json and <stem>.h5)")
    parser.add_argument("--two-bit", help="Reference genome in .2bit (required for 'legacy' and 'bird')")
    parser.add_argument("--seq-len", type=int, required=True,
                        help="Window length to extract around each variant; will be overridden by model input length")
    parser.add_argument("--output", default="-", help="Output TSV (default: stdout)")
    parser.add_argument("--job-size", type=int, default=128, help="Batch size for model.predict")
    parser.add_argument("--twobittofa", default="twoBitToFa", help="Path to UCSC twoBitToFa")
    parser.add_argument("--max-n", type=int, default=-1, help="Optional stop after parsing N items")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = load_model_from_stem(args.model_stem)
    expected_len = model.input_shape[1]
    seq_len = expected_len
    if args.seq_len != expected_len:
        print(
            f"[WARN] Overriding --seq-len {args.seq_len} -> {expected_len} to match model input length",
            file=sys.stderr
        )

    # ----------------- FASTA MODE -----------------
    if args.input_format == "fasta":
        fasta_records = read_fasta(args.input)
        if not fasta_records:
            raise ValueError("No records found in FASTA file")

        out = sys.stdout if args.output == "-" else open(args.output, "wt")
        try:
            print("ID\twindow\tpos\tref\tallele\tallele_type\tprediction", file=out)

            seqs_batch: List[str] = []
            meta_batch: List[Tuple[str, str, int, str, str, str]] = []

            for hdr, seq in fasta_records:
                seq = seq.strip().upper()
                if len(seq) < expected_len:
                    raise ValueError(
                        f"Sequence {hdr!r} length {len(seq)} < model expected length {expected_len}"
                    )
                if len(seq) > expected_len:
                    seq = seq[:expected_len]

                chrom, region_start, region_end = parse_interval_header(hdr)
                window = f"{chrom}:{region_start}-{region_end}"

                for i, ref_base in enumerate(seq):
                    if ref_base not in ALPHABET:
                        logging.warning(
                            f"Skipping position {i} in {hdr!r} due to ambiguous base {ref_base!r}"
                        )
                        continue

                    pos1 = region_start + i

                    for allele in ("A", "C", "G", "T"):
                        alt_seq = seq if allele == ref_base else (seq[:i] + allele + seq[i+1:])
                        allele_type = "ref" if allele == ref_base else "alt"
                        seqs_batch.append(alt_seq)
                        meta_batch.append((hdr, window, pos1, ref_base, allele, allele_type))

                        if len(seqs_batch) >= args.job_size:
                            X = one_hot_batch(seqs_batch)
                            y = np.asarray(model.predict(X, batch_size=len(seqs_batch), verbose=0)).reshape((-1,))
                            for (ID, win, p, r, a, atype), yhat in zip(meta_batch, y):
                                print(f"{ID}\t{win}\tpos={p}\tref={r}\t{a}\t{atype}\t{float(yhat)}", file=out)
                            seqs_batch.clear()
                            meta_batch.clear()

            if seqs_batch:
                X = one_hot_batch(seqs_batch)
                y = np.asarray(model.predict(X, batch_size=len(seqs_batch), verbose=0)).reshape((-1,))
                for (ID, win, p, r, a, atype), yhat in zip(meta_batch, y):
                    print(f"{ID}\t{win}\tpos={p}\tref={r}\t{a}\t{atype}\t{float(yhat)}", file=out)

        finally:
            if out is not sys.stdout:
                out.close()
        return

    # ----------------- VARIANT MODES -----------------
    if not args.two_bit:
        raise ValueError("--two-bit is required when using 'legacy' or 'bird' input_format")

    items: List[Dict[str, Any]] = []
    half = seq_len // 2

    if args.input_format == "legacy":
        cres = load_cres(args.input, max_n=args.max_n)
        for cre in cres:
            for posrec in cre.positions:
                pos1 = posrec.pos
                pos0 = pos1 - 1
                begin = max(0, pos0 - half)
                end = begin + seq_len
                items.append({
                    "chrom": cre.chrom,
                    "cre_id": f"{cre.chrom}:{cre.begin}-{cre.end}",
                    "begin": begin,
                    "end": end,
                    "pos1": pos1,
                    "pos0": pos0,
                    "ref": posrec.ref.upper(),
                    "alleles": [posrec.ref.upper()] + [a.upper() for a in posrec.alleles if a.upper() != posrec.ref.upper()],
                })
                if 0 < args.max_n <= len(items):
                    break
            if 0 < args.max_n <= len(items):
                break

    elif args.input_format == "bird":
        variants = load_bird_variants(args.input, max_n=args.max_n)
        for v in variants:
            pos1 = v.pos
            pos0 = pos1 - 1
            begin = max(0, pos0 - half)
            end = begin + seq_len
            items.append({
                "chrom": v.chrom,
                "cre_id": v.variant_id,
                "begin": begin,
                "end": end,
                "pos1": pos1,
                "pos0": pos0,
                "ref": v.ref.upper(),
                "alt": v.alt.upper(),
                "alleles": [v.ref.upper(), v.alt.upper()],  # <- both ref and alt
            })

    if not items:
        raise RuntimeError("No items to process after parsing input")

    with tempfile.TemporaryDirectory() as tmpd:
        coords_path = os.path.join(tmpd, "coords.txt")
        fasta_path = os.path.join(tmpd, "sequences.fa")

        with open(coords_path, "wt") as fh:
            fh.write("\n".join(f"{it['chrom']}:{it['begin']}-{it['end']}" for it in items) + "\n")

        run_twoBitToFa(args.two_bit, coords_path, fasta_path, args.twobittofa)

        fasta_records = read_fasta(fasta_path)
        if len(fasta_records) != len(items):
            raise RuntimeError(
                f"twoBitToFa output count ({len(fasta_records)}) doesn't match variant count ({len(items)})"
            )

        out = sys.stdout if args.output == "-" else open(args.output, "wt")
        try:
            print("ID\twindow\tpos\tref\tallele\tallele_type\tprediction", file=out)

            seqs_batch: List[str] = []
            meta_batch: List[Tuple[str, str, int, str, str, str]] = []

            for (header, seq), it in zip(fasta_records, items):
                seq = seq.upper()

                if any(c not in ALPHABET for c in seq):
                    logging.warning(
                        f"Skipping {it['chrom']}:{it['begin']}-{it['end']} due to ambiguous base(s) in sequence"
                    )
                    continue

                pos0 = it["pos0"]
                local = pos0 - it["begin"]
                if local < 0 or local >= len(seq):
                    raise ValueError(
                        f"Position {it['pos1']} outside extracted window {it['chrom']}:{it['begin']}-{it['end']}"
                    )

                ref_base = seq[local]
                listed_ref = it["ref"]
                if ref_base != listed_ref:
                    raise ValueError(
                        f"Ref mismatch at {it['chrom']}:{it['pos1']}: genome={ref_base} vs listed ref={listed_ref}"
                    )

                window_str = f"{it['chrom']}:{it['begin']}-{it['end']}"

                for allele in it["alleles"]:
                    allele = allele.upper()
                    if allele not in ALPHABET:
                        logging.warning(
                            f"Skipping non-ACGT allele {allele!r} at {it['chrom']}:{it['pos1']}"
                        )
                        continue

                    mut_seq = seq if allele == ref_base else (seq[:local] + allele + seq[local+1:])
                    allele_type = "ref" if allele == ref_base else "alt"

                    seqs_batch.append(mut_seq)
                    meta_batch.append((it["cre_id"], window_str, it["pos1"], ref_base, allele, allele_type))

                    if len(seqs_batch) >= args.job_size:
                        X = one_hot_batch(seqs_batch)
                        y = np.asarray(model.predict(X, batch_size=len(seqs_batch), verbose=0)).reshape((-1,))
                        for (ID, window, p, r, a, atype), yhat in zip(meta_batch, y):
                            print(f"{ID}\t{window}\tpos={p}\tref={r}\t{a}\t{atype}\t{float(yhat)}", file=out)
                        seqs_batch.clear()
                        meta_batch.clear()

            if seqs_batch:
                X = one_hot_batch(seqs_batch)
                y = np.asarray(model.predict(X, batch_size=len(seqs_batch), verbose=0)).reshape((-1,))
                for (ID, window, p, r, a, atype), yhat in zip(meta_batch, y):
                    print(f"{ID}\t{window}\tpos={p}\tref={r}\t{a}\t{atype}\t{float(yhat)}", file=out)

        finally:
            if out is not sys.stdout:
                out.close()

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")
    main()