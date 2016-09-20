#!/bin/bash
process_number=3
Ip=("0.0.0.0" "0.0.0.0")
for ip in ${Ip[@]}
do
    ssh worker@$ip rm /home/worker/xiaoshu/logistic-regression-ftrl-mpi/train
    scp train worker@$ip:/home/worker/xiaoshu/logistic-regression-ftrl-mpi/.
done
mpirun -f ../hosts -np $process_number ./train ftrl 10 10 0.0 0.1 1.0 0.001 0.0 ./data/agaricus.txt.train ./data/agaricus.txt.test
