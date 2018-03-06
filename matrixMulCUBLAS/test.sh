#!/bin/bash

nvidia-smi -q -d POWER -lms 100 -f gpu_power 2>&1 1>/dev/null & smi_pid=$!

./matrixMulCUBLAS

kill ${smi_pid}

grep "Power Draw" < gpu_power | sed 's/[^0-9.]*//g' > gpu_power_parsed


