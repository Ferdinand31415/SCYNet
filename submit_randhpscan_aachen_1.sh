#!/usr/bin/bash
 
### Job name
#BSUB -J randomhyperscan

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
#BSUB -o /home/fe918130/shelloutputs/randhyperscan.%J.%I
 
### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute,
### that means for 80 minutes you could also use this: 1:20
#BSUB -W 120
 
### Request memory you need for your job in TOTAL in MB
#BSUB -M 10000
#####1024


### q (we) or a (test)
#BSUB -q gpu

### -R resource requirements
#BSUB -R kepler

 

module switch intel gcc
module load python
module load cuda/80
export LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH


### Change to the work directory
cd /home/fe918130/SCYNET
export CUDA_VISIBLE_DEVICES=1
python pmssm.py
