#!/bin/bash

#SBATCH --account=arupbio-kp
#SBATCH --partition=arup-kp
#SBATCH --nodes=1
#SBATCH --time=3-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com



# Cant activate a conda env non-interactively, so just set the  python binary
# to the right spot - seems to work
PYTHON=$HOME/miniconda3/envs/jv2/bin/python

ds2s=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/dnaseq2seq/main.py

#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/multindel_novaav_conf.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_multindel_conf.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_multindel_splits_chrsE.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_bigvars_conf.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_lcbig_sus_chrs21and22.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_lcbig_sus_5more_chrsE.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/decoder_fpfn_chr1_conf.yaml

DEST=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_depth200_5more_chrsE

BATCH_SIZE=512

mkdir -p $DEST
cp $CONF $DEST/

$PYTHON $ds2s pregen -c $CONF -d $DEST --threads 6 --batch-size $BATCH_SIZE > $DEST/stdout.log 

