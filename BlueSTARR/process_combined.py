# process A549 extra CGs data

import pandas as pd
from FastaWriter import FastaWriter
from tqdm import tqdm
import sys
import gzip
import os

import argparse


lib_size = pd.read_table('/work/igvf-pm/A549/extra_GCs/A549.starrseq.GR_agonists.lib_sizes.txt', names=['size','rep'], sep=' ')
lib_size_dict = lib_size.set_index('rep')['size'].to_dict()

def normalize(df_count, label):
    c = 10**6

    lib_size = pd.read_table('/work/igvf-pm/A549/extra_GCs/A549.starrseq.GR_agonists.lib_sizes.txt', names=['size','rep'], sep=' ')
    lib_size_dict = lib_size.set_index('rep')['size'].to_dict()

    df_count['input_rep1'] = df_count['input_rep1']/lib_size_dict['Input1']*c
    df_count['input_rep2'] = df_count['input_rep2']/lib_size_dict['Input2']*c
    df_count['input_rep3'] = df_count['input_rep3']/lib_size_dict['Input3']*c
    df_count['input_rep4'] = df_count['input_rep4']/lib_size_dict['Input4']*c
    df_count['input_rep5'] = df_count['input_rep5']/lib_size_dict['Input5']*c
    df_count['A549_'+label+'_rep2'] = df_count['A549_'+label+'_rep2']/lib_size_dict['TFX2_'+label]*c
    df_count['A549_'+label+'_rep3'] = df_count['A549_'+label+'_rep3']/lib_size_dict['TFX3_'+label]*c
    df_count['A549_'+label+'_rep4'] = df_count['A549_'+label+'_rep4']/lib_size_dict['TFX4_'+label]*c
    df_count['A549_'+label+'_rep5'] = df_count['A549_'+label+'_rep5']/lib_size_dict['TFX5_'+label]*c

    return df_count

def normalize_counts(input_dir,output_dir,labels):

    input_path = input_dir+'A549.'+label+'.combined.input_and_output.w600s50.log2FC.gt_200.sequence.final.txt.gz' # shreshold=200, seqlen=600
    df_combined = pd.read_csv(input_path, sep = '\t', compression="gzip")

    output_path = output_dir+label
    os.makedirs(output_path, exist_ok=True)

    # process lines to fhave the format of >17698785 /coord=chr1:138900-139200
    # CCTGTAGACGCTGACAGGAGGCAGGAGCTGGGCCTGGACAGGTCAACTTGAGGAGATTTT

    ls_chr = df_combined['chrom']
    ls_start = df_combined['start']
    ls_end = df_combined['end']
    ls_seq = df_combined['sequence']

    fastaWriter=FastaWriter()
    # output_fasta = gzip.open(output_path+label+"-all.fasta.gz","wt") # slow
    output_fasta = open(output_path+"/all.fasta","wt") 

    for i in range(len(df_combined)):
        defline = '>'+str(i)+' /coord='+str(ls_chr[i])+':'+str(ls_start[i])+'-'+str(ls_end[i])
        assert len(ls_seq[i]) == 600
        fastaWriter.addToFasta(defline,ls_seq[i],output_fasta)

    # process counts to have the format
    # DNA=5    RNA=4
    # 1   2   3   4   5   6   7    8    9

    count_cols = ['input_rep1', 'input_rep2', 'input_rep3', 'input_rep4', 'input_rep5'] + \
                    ['A549_'+label+'_rep'+str(i) for i in range(2,6)]
    df_count = df_combined[count_cols].astype(float)
    # df_count = normalize(df_count, label)

    output_count = gzip.open(output_path+'/all-counts.txt.gz',"wt")
    header = "DNA=5\tRNA=4\n"
    print(header,end="",file=output_count)
    df_count.to_csv(output_count, sep='\t', index=False, header=None) 

def main():
    arg = argparse.ArgumentParser(description="<add_description")

    arg.add_argument("--input", "-i", required=True, help="path to input .txt.gz file which has the combined input and output.")
    arg.add_argument("--library-sizes", "-lib", required=True, help="Path to the text file with the output and input library sizes")
    


    input_path = input_dir+'A549.'+label+'.combined.input_and_output.w600s50.log2FC.gt_200.sequence.final.txt.gz' # shreshold=200, seqlen=600
    df_combined = pd.read_csv(input_path, sep = '\t', compression="gzip")

    output_path = output_dir+label
    os.makedirs(output_path, exist_ok=True)


    
if(len(sys.argv)!=4):
    exit(ProgramName.get()+" <input-dir:/.../> <output-dir:/.../> <label:A,B,C>\n")
(input_dir,output_dir,labels)=sys.argv[1:]

labels = labels.split(',') if ',' in labels else [labels]
for label in labels:
    main(input_dir,output_dir,label)

# command line:
# python /work/igvf-pm/A549/extra_GCs/process_combined.py /work/igvf-pm/alex_b/starrseq_A549_extra_GCs/ /work/igvf-pm/A549/extra_GCs/ AZD2906,AZD9567,CORT108297,CpdA,GW870086,Hydrocortisone,Mapracorat,RU486,ZK216348
# python /work/igvf-pm/A549/extra_GCs/process_combined.py /work/igvf-pm/alex_b/starrseq_A549_extra_GCs/ /work/igvf-pm/A549/extra_GCs/600bp/data-normalized/ AZD2906,AZD9567,CORT108297,CpdA,GW870086,Hydrocortisone,Mapracorat,RU486,ZK216348