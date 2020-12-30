#!/bin/bash
read -p "Specified device ?= " device;
read -p "Specified models(SRCNN, ESPCN, RDDBNet, EDSR etc.) ?= " models;
for m in $models; do
    for up in 4 8 16; do
        CUDA_VISIBLE_DEVICE=$device python ./src/trainCas.py --SRModel $m --up $up;
    done
done
echo DONE...
