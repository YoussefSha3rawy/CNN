#!/bin/bash
#SBATCH --job-name CNN                             # Job name
#SBATCH --partition=prigpu                         # Select the correct partition.
#SBATCH --nodes=1                                  # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=4                          # Use 4 cores, most of the procesing happens on the GPU
#SBATCH --mem=90GB                                 # Expected ammount CPU RAM needed (Not GPU Memory)
#SBATCH --time=72:00:00                            # Expected ammount of time to run Time limit hrs:min:sec
#SBATCH --gres=gpu:1                               # Use one gpu.
#SBATCH -e logs/%x_%j.e                            # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o logs/%x_%j.o                            
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adfx751@city.ac.uk

#Enable modules command

source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu

python --version
nvidia-smi

wandb login $WANDB_API_KEY --relogin
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
python3 cnn.py
