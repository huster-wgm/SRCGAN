#!/bin/bash
read -p "Specified device ?= " device;
read -p "Specified models(ESPCN, RDDBNet, EDSR etc.) ?= " models;
for m in $models; do
    for up in 2; do
        CUDA_VISIBLE_DEVICE=$device python ./src/trainCas.py --SRModel $m --up $up;
        for ep in 25 50; do
            # e.g., GA = EPSCN_A2C_x2_0005.pth, GB = RDDBNet_C2B_x2_0005.pth
            GA=$m\_A2C_x$up\_00$ep.pth;
            GB=ResDeconv_C2B_x$up\_00$ep.pth;
            CUDA_VISIBLE_DEVICE=$device python ./src/testCas.py \
                --netGA ./checkpoints/$GA \
                --netGB ./checkpoints/$GB;
        done
    done
done
echo DONE...
