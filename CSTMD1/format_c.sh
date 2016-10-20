#!/bin/bash
~/Downloads/astyle/build/gcc/bin/astyle Cuda/src/*.cu
~/Downloads/astyle/build/gcc/bin/astyle Cuda/src/*.h
mkdir Cuda/src/origs
mv Cuda/src/*.orig Cuda/src/origs
