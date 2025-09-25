#!/bin/bash

cuobjdump -sass ./fft > fft.sass

compute-sanitizer --tool memcheck ./fft
compute-sanitizer --tool racecheck ./fft

ncu --set full -o fft -f ./fft