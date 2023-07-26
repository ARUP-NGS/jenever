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

ds2s=/uufs/chpc.utah.edu/common/home/u0379426/src/jovian/dnaseq2seq/main.py

#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/multindel_novaav_conf.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_multindel_conf.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_multindel_splits_chrsE.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_bigvars_conf.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_lcbig_sus_chrs21and22.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_lcbig_sus_5more_chrsE.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/decoder_fpfn_chr1_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/jovian/pregen_confs/wgs_lcbigmap_chrs21and22.yaml

DEST=/uufs/chpc.utah.edu/common/home/arup-storage4/brendan/pregen/pregen_lcbigmap_n1k_chrs21and22

BATCH_SIZE=512

mkdir -p $DEST
cp $CONF $DEST/

$PYTHON $ds2s pregen -c $CONF -d $DEST --threads 28 --batch-size $BATCH_SIZE > $DEST/stdout.log 

