#!/bin/bash

MASTER_ADDR=$1
MASTER_PORT=$2
RUN_NAME=$3
RUN_NOTES="$4"

#VAL_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/pregen/pregen_lcbigmap_d150_chrs21and22
VAL_DIR=/scratch/general/vast/u0379426/pregen_lcbigmap_d150_chrs21and22
PREGEN_DIR=/scratch/general/vast/u0379426/lcbigmap2x
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/pregen/fpfns_moresupp


RUNCMD="jovian/dnaseq2seq/main.py train \
    -d $PREGEN_DIR \
    --val-dir $VAL_DIR \
    -n 500 \
    --batch-size 200 \
    --learning-rate 0.00005 \
    --checkpoint-freq 10 \
    -i /uufs/chpc.utah.edu/common/home/u0379426/storage/variant_transformer_runs/200M_lcbig2x/200M_lcbig2x_epoch70.model \
    -o ${RUN_NAME}.model \
    --threads 16 \
    --max-decomp-batches 8 \
    --samples-per-epoch 500000 \
    --wandb-run-name $RUN_NAME"

echo "Full run cmd: $RUNCMD"
echo "Master addr: $MASTER_ADDR, master port: $MASTER_PORT"

export ENABLE_COMET=1

$HOME/miniconda3/envs/jv2/bin/torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=$SLURM_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $RUNCMD

