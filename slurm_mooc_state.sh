#!/bin/sh
#SBATCH -J 8-mooc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100g
#SBATCH -c 10
#SBATCH -t 24:00:00
#SBATCH -o /mnt/beegfs/home/gauthier/logs/%x.out
#SBATCH -e /mnt/beegfs/home/gauthier/logs/%x.err
#SBATCH --mail-user=vincent.gauthier@telecom-sudparis.eu
#SBATCH --mail-type=ALL

cd /mnt/beegfs/home/gauthier/JODIE
source env/bin/activate
echo ""
echo ""
echo "###########################################################################"
echo "Python interpreter: $(which python)"
echo "running python script: train_evaluate_mooc_state.py"
echo "###########################################################################"
echo ""
echo ""

python -u train_evaluate_mooc_state.py
