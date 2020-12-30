#!/bin/bash
read -p "Specified device ?= " device;
for m in ESPCN SRCNN RDDBNet EDSR; do
    for up in 4 8 16; do
        for ep in 25 50; do
            # e.g., GA = EPSCN_A2C_x2_0005.pth, GB = RDDBNet_C2B_x2_0005.pth
            GA=$m\_A2C_x$up\_00$ep.pth;
            GB=ResDeconv_C2B_x$up\_00$ep.pth;
            echo "GA => " $GA "; GB => " $GB; 
            CUDA_VISIBLE_DEVICE=$device python ./src/visCas.py \
                --netGA ./checkpoints/$GA \
                --netGB ./checkpoints/$GB \
                --threshold 25.0 ;
        done
    done
done
echo DONE...