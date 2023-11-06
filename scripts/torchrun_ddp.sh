#!/bin/bash

MASTER_ADDR=$1
MASTER_PORT=$2
RUN_NAME=$3
RUN_NOTES="$4"

#VAL_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/pregen/pregen_lcbigmap_d150_chrs21and22
VAL_DIR=/scratch/general/vast/u0379426/pregen_lcbigmap_d150_chrs21and22
PREGEN_DIR=/scratch/general/vast/u0379426/lcbigmap2x


RUNCMD="jovian/dnaseq2seq/main.py train \
    -d $PREGEN_DIR \
    --val-dir $VAL_DIR \
    -n 500 \
    --batch-size 128 \
    --learning-rate 0.00003 \
    -i /uufs/chpc.utah.edu/common/home/u0379426/storage/variant_transformer_runs/100M_lcbig2x_again_cont/100M_lcbig2x_again_cont_epoch170.model \
    --checkpoint-freq 2 \
    -o ${RUN_NAME}.model \
    --threads 16 \
    --max-decomp-batches 8 \
    --samples-per-epoch 500000 \
    --wandb-run-name $RUN_NAME"

echo "Full run cmd: $RUNCMD"
echo "Master addr: $MASTER_ADDR, master port: $MASTER_PORT"


export ENABLE_COMET=1
export COMET_GIT_DIRECTORY=jovian/

$HOME/miniconda3/envs/jv2/bin/torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=$SLURM_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $RUNCMD
