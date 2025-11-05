#!/usr/bin/env python
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import gzip
import os

fasta_files = {
    "Train": "/hpc/home/rv103/igvf/revathy/sequences/K562/1000bp/train.fasta.gz",
    "Validation": "/hpc/home/rv103/igvf/revathy/sequences/K562/1000bp/validation.fasta.gz",
    "Test": "/hpc/home/rv103/igvf/revathy/sequences/K562/1000bp/test.fasta.gz"
}

expected_length = 1000

for label, filepath in fasta_files.items():
    if not os.path.exists(filepath):
        print(f"{label} file not found: {filepath}")
        continue

    print(f"\n Filtering {label} file: {os.path.basename(filepath)}")

    valid_records = []

    with gzip.open(filepath, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if len(record.seq) == expected_length:
                valid_records.append(record)

    print(f"Kept {len(valid_records)} sequences with length = {expected_length}")

    # Overwrite the original file with filtered content
    with gzip.open(filepath, "wt") as out_handle:
        SeqIO.write(valid_records, out_handle, "fasta")

    print(f"Updated file written: {filepath}")
