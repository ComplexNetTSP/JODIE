#!/bin/sh
#SBATCH -J JODIE
#SBATCH --nodes=1
#SBATCH --mem=100g
#SBATCH -c 10
#SBATCH -t 24:00:00
#SBATCH -o /mnt/beegfs/home/gauthier/logs/%x.out
#SBATCH -e /mnt/beegfs/home/gauthier/logs/%x.err
#SBATCH --mail-user=vincent.gauthier@telecom-sudparis.eu
#SBATCH --mail-type=ALL

cd /mnt/beegfs/home/gauthier/JODIE
source env/bin/activate 
python train_evaluate_wikipedia_state.py