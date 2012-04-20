#!/bin/bash
for (( c=2; c<=2097152; c=c*2 ))
do
#    echo "Command is ./fft $c >>output.csv"
    ./fft $c >>fftw_output.csv
    #./cufft $c >>m2090_output.csv         
done
