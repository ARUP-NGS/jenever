#!/bin/bash

set -x

REF_GENOME=/uufs/chpc.utah.edu/common/home/u0379426/vast/ref/human_g1k_v37_decoy_phiXAdaptr.fasta.gz
DEST_DIR=$HOME/data/variant-transformer


GIABROOT=/home/22319/data/variant-transformer/

./splitbed.py $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/bed/HG001_GRCh37_1_22_v4.2.1_benchmark.bed > ${DEST_DIR}/giab_labelled_beds/HG001_split.bed
./splitbed.py $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/bed/HG002_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed > ${DEST_DIR}/giab_labelled_beds/HG002_split.bed
./splitbed.py $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/bed/HG003_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed > ${DEST_DIR}/giab_labelled_beds/HG003_split.bed
./splitbed.py $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/bed/HG004_GRCh37_1_22_v4.2.1_benchmark_noinconsistent.bed > ${DEST_DIR}/giab_labelled_beds/HG004_split.bed
./splitbed.py $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/bed/HG005_GRCh37_1_22_v4.2.1_benchmark.bed > ${DEST_DIR}/giab_labelled_beds/HG005_split.bed
./splitbed.py $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/bed/HG006_GRCh37_1_22_v4.2.1_benchmark.bed > ${DEST_DIR}/giab_labelled_beds/HG006_split.bed
./splitbed.py $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/bed/HG007_GRCh37_1_22_v4.2.1_benchmark.bed > ${DEST_DIR}/giab_labelled_beds/HG007_split.bed


 for VCF in $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/vcf/HG001_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/vcf/HG002_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/vcf/HG003_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/vcf/HG004_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/vcf/HG005_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/vcf/HG006_GRCh37_1_22_v4.2.1_benchmark.vcf.gz \
    $GIABROOT/GIAB_NISTv4.2.1_2023-10-26/vcf/HG007_GRCh37_1_22_v4.2.1_benchmark.vcf.gz; do
        PREFIX=$(basename $VCF | cut -d "_" -f1)
        echo $PREFIX;
        ./assignclasses.py $VCF $GIABROOT/GIAB_stratifications/GRCh37_AllTandemRepeatsandHomopolymers_slop5.bed.gz $GIABROOT/GIAB_stratifications/GRCh37_alllowmapandsegdupregions.bed.gz ${DEST_DIR}/giab_labelled_beds/${PREFIX}_split.bed > ${DEST_DIR}/giab_labelled_beds/${PREFIX}_split_labels.bed
    done




