#!/usr/bin/env bash

# https://github.com/btgraham/SparseConvNet/tree/master/SparseConvNet/Data/kaggleDiabeticRetinopathy
rm train/492_*
cp test/25313_left.jpeg test/25313_right.jpeg
cp test/27096_left.jpeg test/27096_right.jpeg

for a in $(grep -v image trainLabels.csv|grep -v ^492_);
    do b=echo $a|cut -d , -f 1;
        c=echo $a|cut -d , -f 2;
        ln -s ../../train/$b.jpeg train01234/$c/$b.jpeg;
    done

for a in $(grep Private retinopathy_solution.csv);
    do b=echo $a|cut -d , -f 1;
        c=echo $a|cut -d , -f 2;
        ln -s ../../test/$b.jpeg testPrivate01234/$c/$b.jpeg;
    done

for a in $(grep Public retinopathy_solution.csv);
    do b=echo $a|cut -d , -f 1;
        c=echo $a|cut -d , -f 2;
        ln -s ../../test/$b.jpeg testPublic01234/$c/$b.jpeg;
    done
