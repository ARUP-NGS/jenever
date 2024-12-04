#!/usr/bin/env python


import sys
import pysam
from collections import defaultdict
import gzip
from quicksect import IntervalTree
#from intervaltree import IntervalTree
import random

"""
Given a 'split' BED file (where every region has the same length, usually 300bp), generate a fourth column
of the BED file that contains a label that can be used for upsampling / downsampling

Right now we do this based on the presence of variants, so regions are labeled according to whether or not
they contain snvs, indels, etc etc. Anything goes, really

"""


def buildforest(bedpath, minsize=0):
    forest = defaultdict(IntervalTree)
    if bedpath.endswith(".gz"):
        fh = gzip.open(bedpath, mode='rt')
    else:
        fh = open(bedpath)
    for line in fh:
        if len(line.strip()) == 0 or line.startswith("#"):
            continue
        toks = line.split("\t")
        chrom = toks[0]
        start = int(toks[1])
        end = int(toks[2])
        size = end - start
        if size >= minsize:
            forest[chrom].add(start, end) # For quicksect, use add() not addi()
    return forest



sys.stderr.write(f"Loading variants from {sys.argv[1]}\n")
sys.stderr.flush()
vcf = pysam.VariantFile(sys.argv[1])

sys.stderr.write("Loading forast from " + sys.argv[2] + "\n")
sys.stderr.flush()
forest = buildforest(sys.argv[2])

mappability_forest = None
mappability_count = 0
if len(sys.argv) > 3:
    sys.stderr.write(f"Loading additional interval set from {sys.argv[3]}\n")
    sys.stderr.flush()
    mappability_forest = buildforest(sys.argv[3], minsize=100)


for line in open(sys.argv[4]):
    variants = []
    toks = line.split("\t")
    start = int(toks[1])
    end = int(toks[2])

    chrom = toks[0]
    variants = list(vcf.fetch(chrom, start, end))
    intervals = list(forest[chrom].search(start, end))  # quicksect version
    #intervals = list(forest[chrom].overlap(start, end))  # intervaltree version
    interval_count = len(intervals)

    if mappability_forest:
        #mappability_count = len(list(mappability_forest[chrom].overlap(start, end))) # intervaltree
        mappability_count = len(list(mappability_forest[chrom].search(start, end)))  # quicksect


    snv_count = len([v for v in variants if len(v.ref) == 1 and len(v.alts[0]) == 1])
    del_count = len([v for v in variants if len(v.ref) > 1 and len(v.alts[0]) == 1])
    ins_count = len([v for v in variants if len(v.ref) == 1 and len(v.alts[0]) > 1])
    multi_count = len([v for v in variants if len(v.alts) > 1])
 
    if variants:
        var_size = max( max(len(v.ref), max(len(a) for a in v.alts)) for v in variants)
    else:
        var_size = 0

    if del_count and ins_count:
        label = "ins-del"
    elif (ins_count > 0 or del_count > 0) and snv_count > 0:
        label = "indel-snv"
    elif multi_count:
        label = "multi"
    elif del_count:
        label = "del"
    elif ins_count:
        label = "ins"
    elif snv_count:
        label = "snv"
    else:
        label = "tn"
    found = False
    for i in intervals:
        if any(i.start < v.pos < i.end for v in variants):
            found = True

    if var_size > 10:
        label = label + "-big"

    if found:
        label = label + "-lc"

    if mappability_count:
        label = label + "-map"

    if label == 'tn' and random.random() < 0.5:
        continue

    toks = '\t'.join(line.strip().split('\t')[0:3])
    print(f"{toks}\t{label}")

