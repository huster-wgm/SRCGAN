#!/bin/bash
sroot=./result;
troot=../geoseg/dataset/Sat2Aer/img;
for m in ESPCN SRCNN; do
    for up in 2; do
        for ep in 25 50; do
            # e.g., src = A_EDSR_x4_0025, tar = EDSRx2@ep25
            src=A_$m\@G2LAB_x$up\_00$ep;
            tar=$m\@G2LABx$up@ep$ep;
            echo "mv from $src to $tar";
            if [[ -d $troot/$tar ]]
            then
                echo "remove existing $tar";
                rm -rf $troot/$tar;
            fi
            mv ./result/$src $troot/$tar;
        done
    done
done