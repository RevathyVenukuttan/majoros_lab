#!/bin/bash

indir="/hpc/home/rv103/igvf/revathy/Blueplots/predictions/Scaled_15/mse"
outfile="/hpc/home/rv103/igvf/revathy/Blueplots/predictions/Scaled_15/mse/K562_0.15_avg.txt"
files=($indir/*.txt)
awk '
BEGIN { OFS="\t" }    # ensure tab-separated output
{
    # Use FNR as row index (row number within each file)
    row[FNR] = $1 OFS $2 OFS $3 OFS $4 OFS $5
    sum[FNR] += $6
    count[FNR]++
}
END {
    for (i=1; i<=FNR; i++) {
        avg = sum[i] / count[i]
        print row[i], avg
    }
}' "${files[@]}" > "$outfile"

echo "Averaged file written to $outfile"
