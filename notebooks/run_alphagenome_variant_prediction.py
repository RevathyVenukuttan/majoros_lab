""" This script runs variant prediction using alphagenome """

###################################
# set environment
#----------------------------------

import numpy  as np
import pandas as pd
import os
import argparse

import time
import random
import grpc
from grpc import StatusCode

from alphagenome.data   import genome
from alphagenome.models import dna_client, variant_scorers


###################################
# Helper function
#----------------------------------

def score_with_retry(
    dna_model, interval, variant, scorers,
    max_retries=6, base=2.0, cap=60.0, min_delay=0.3):
    """Call score_variant with pacing to avoid potential quota hit error"""
    
    # polite pacing so one job doesn't burn the minute quota
    time.sleep(min_delay)

    # query using score variant; retry max_retries times at max
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            return dna_model.score_variant(
                interval=interval,
                variant=variant,
                variant_scorers=scorers
            )
        except grpc.RpcError as e:
            # retry after a short delay; increase delay time when retry
            if e.code() == StatusCode.RESOURCE_EXHAUSTED and attempt < max_retries:
                
                # add a small randomness to avoid all concurrent jobs query at the same time
                # cap the delay time
                num_time_sleep = min(cap, delay) * (1.0 + 0.25*random.random())

                # dealy the API call
                time.sleep(num_time_sleep)

                # increase the delay time exponentially
                delay *= base
                continue
            raise  # raise other errors or if retries exhausted

###################################
# Main function
#----------------------------------

def main(args):
    
    ###################################
    # load model
    #----------------------------------
    # check API key
    assert "ALPHAGENOME_API_KEY" in os.environ, "Missing API key"
    
    # load model
    dna_model = dna_client.create(os.environ["ALPHAGENOME_API_KEY"])

    ###################################
    # streaming output
    #----------------------------------
    is_first_write = True
    
    with open(args.input, "r") as file:
        for line in file:
            
            ###################################
            # Parse/Process each line
            #----------------------------------
            # quick quard, skip comment
            if not line.strip():
                continue
            
            # parse each line
            txt_region, num_position, txt_variant_reference_base, txt_variant_alternate_base = line.strip().split("\t")
    
            # parse region coordinates
            txt_chrom_name,  txt_coords    = txt_region.split(":")
            num_chrom_start, num_chrom_end = txt_coords.split("-")
            
            # convert typing
            num_position    = int(num_position)
            num_chrom_start = int(num_chrom_start)
            num_chrom_end   = int(num_chrom_end)
            
            ###################################
            # Set genomic variant and interval
            #----------------------------------
            # define genome variant
            num_variant_position_0base = num_position
            num_variant_position_1base = num_variant_position_0base + 1
            
            variant = genome.Variant(
                chromosome = txt_chrom_name,
                position   = num_variant_position_1base,
                reference_bases = txt_variant_reference_base,
                alternate_bases = txt_variant_alternate_base,
            )
    
            # Specify length of sequence around variant to predict:
            # length options: ["2KB", "16KB", "100KB", "500KB", "1MB"]
            txt_sequence_length = args.seqlen  
            num_sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
                f'SEQUENCE_LENGTH_{txt_sequence_length}'
            ]
    
            # set interval centering at the variant
            interval = variant.reference_interval.resize(num_sequence_length)
    
            # pick scorers
            #list(variant_scorers.RECOMMENDED_VARIANT_SCORERS.values())
            #lst_variant_scorer = [variant_scorers.RECOMMENDED_VARIANT_SCORERS["ATAC"]]
            #lst_variant_scorer = [variant_scorers.RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"]]
            lst_variant_scorer = [variant_scorers.RECOMMENDED_VARIANT_SCORERS[args.scorer]]

            # run prediction (original code)
            #variant_scores = dna_model.score_variant(
            #    interval = interval,
            #    variant  = variant,
            #    variant_scorers = lst_variant_scorer
            #)

            # Update: run prediction with retry/pacing
            # if exceed quota, retry; output errors if unsucceed
            try:
                variant_scores = score_with_retry(
                    dna_model, interval, variant, lst_variant_scorer,
                    min_delay=args.min_delay
                )
            except Exception as e:
                # sidecar errors file, keep going
                with open(args.output + ".errors", "a") as ferr:
                    ferr.write(f"{txt_region}\t{num_position}\t{txt_variant_reference_base}\t{txt_variant_alternate_base}\t{repr(e)}\n")
                continue

            # convert results into a dataframe
            dat_scores = variant_scorers.tidy_scores(variant_scores)
            
            # Skip if nothing came back
            if dat_scores is None or getattr(dat_scores, "empty", False):
                with open(args.output + ".errors", "a") as ferr:
                    ferr.write(
                        f"{txt_region}\t{num_position}\t{txt_variant_reference_base}\t"
                        f"{txt_variant_alternate_base}\tNO_SCORES\n"
                    )
                continue
                
            # apply biosample filter only when requested
            if args.biosample:
                dat_scores = dat_scores.loc[dat_scores["biosample_name"] == args.biosample]
                if dat_scores.empty:
                    # nothing for this biosample-add log and skip
                    with open(args.output + ".errors", "a") as ferr:
                        ferr.write(
                            f"{txt_region}\t{num_position}\t{txt_variant_reference_base}\t"
                            f"\t{txt_variant_alternate_base}\tNO_SCORES_FOR_{args.biosample}\n"
                        )
                    continue
                    
            # arrange results
            dat = dat_scores
            dat = dat.assign(
                Region   = txt_region,
                Position = num_variant_position_0base,
                Ref      = txt_variant_reference_base,
                Alt      = txt_variant_alternate_base,
                Length   = num_sequence_length,
                Scorer   = args.scorer
            )  
            
            # write to file (open in append mode, no index, write column names only on the very first write)
            dat.to_csv(args.output, sep="\t", index=False, mode="a", header=is_first_write)
            is_first_write = False

if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description="AlphaGenome variant scoring")
    parser.add_argument("-i", "--input",  required=True,
                        help="Input TSV file with columns: Region, Position, Ref, Alt")
    parser.add_argument("-o", "--output", required=True,
                        help="Output TSV file for predictions")
    parser.add_argument("-s", "--seqlen", default="2KB",
                        choices=["2KB", "16KB", "100KB", "500KB", "1MB"],
                        help="Sequence length around variant")
    parser.add_argument("-c", "--scorer", default="RNA_SEQ",
                        choices=list(variant_scorers.RECOMMENDED_VARIANT_SCORERS.keys()),
                        help="Which variant scorer to use")
    parser.add_argument("-b", "--biosample", default=None, 
                        help="Optional biosample filter, e.g. K562")
    parser.add_argument("--min-delay", type=float, default=0.3,
                        help="Seconds to sleep after each request (client-side pacing).")
    args = parser.parse_args()

    # execute main
    main(args)
    

        