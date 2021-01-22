#!/bin/bash
read -p "Specified device ?= " device;
read -p "Specified models(SRCNN or SRDN) ?= " models;
for m in $models; do
    for up in 2 4 8; do
#         CUDA_VISIBLE_DEVICE=$device python ./src/trainCasConstLAB.py --SRModel $m --up $up;
        for ep in 25 50; do
            # e.g., GA = EPSCN_A2C_x2_0005.pth, GB = RDDBNet_C2B_x2_0005.pth
            GA=$m\@G2LAB_A2C_x$up\_00$ep.pth;
            GB=ResDeconv@G2LAB_C2B_x$up\_00$ep.pth;
            CUDA_VISIBLE_DEVICE=$device python ./src/testCasConstLAB.py \
                --netGA ./checkpoints/$GA \
                --netGB ./checkpoints/$GB;
        done
    done
done
echo DONE...
