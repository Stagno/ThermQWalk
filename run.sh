#!/bin/sh
# for verbose compilation output use nvcc -lineinfo -keep -keep-dir ./intermediate -src-in-ptx -use_fast_math --resource-usage -arch=sm_35 --std=c++11 -maxrregcount=48  --machine 64 -cudart static --expt-relaxed-constexpr -o ./cuThermQWalk ./ThermQWalk.cu -Xptxas -warn-spills\
echo "Compiling and starting cuThermQWalk with argument $1"
nvcc -lineinfo -keep -keep-dir ./intermediate -src-in-ptx -use_fast_math -arch=sm_35 --std=c++11 -maxrregcount=48  --machine 64 -cudart static --expt-relaxed-constexpr -o ./cuThermQWalk ./ThermQWalk.cu\
&&\
./cuThermQWalk $1
