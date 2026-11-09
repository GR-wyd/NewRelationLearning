#!/bin/bash
#SBATCH -J ours500
#SBATCH -o /gpfs/share/home/2201110046/guorurui/Demo/runs/demo/logs/ours500-%j.out
#SBATCH --cpus-per-task=7
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 7-00:00:00
#SBATCH -p GPUA800

set -euo pipefail

cd /gpfs/share/home/2201110046/guorurui/Demo
source /gpfs/share/home/2201110046/anaconda3/etc/profile.d/conda.sh
conda activate qwen-cu128
export PYTHONNOUSERSITE=1

python validate_ours_500.py "$@"
