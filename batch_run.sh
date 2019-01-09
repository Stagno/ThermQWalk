#!/bin/sh

STARTTIME=$(date +%s)
sed -i '11s/.*/#define DISORDER_PARAM (1.0\/4.0)/' input_parameters.cuh
./run.sh 1over4
ENDTIME=$(date +%s)
echo "Seconds taken: $(($ENDTIME - $STARTTIME))"

STARTTIME=$(date +%s)
sed -i '11s/.*/#define DISORDER_PARAM (1.0\/2.0)/' input_parameters.cuh
./run.sh 1over2
ENDTIME=$(date +%s)
echo "Seconds taken: $(($ENDTIME - $STARTTIME))"

STARTTIME=$(date +%s)
sed -i '11s/.*/#define DISORDER_PARAM (1.0\/1.5)/' input_parameters.cuh
./run.sh 1over1.5
ENDTIME=$(date +%s)
echo "Seconds taken: $(($ENDTIME - $STARTTIME))"

STARTTIME=$(date +%s)
sed -i '11s/.*/#define DISORDER_PARAM (1.0\/1.0)/' input_parameters.cuh
./run.sh 1over1
ENDTIME=$(date +%s)
echo "Seconds taken: $(($ENDTIME - $STARTTIME))"
