#!/bin/bash
#SBATCH -J demo
#SBATCH -o /gpfs/share/home/2201110046/guorurui/Demo/runs/demo/logs/%x-%j.out
#SBATCH --cpus-per-task=7
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 7-00:00:00
#SBATCH -p GPUA800

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: sbatch run_demo.slurm <pipeline command> [arguments...]" >&2
    exit 2
fi

cd /gpfs/share/home/2201110046/guorurui/Demo
source /gpfs/share/home/2201110046/anaconda3/etc/profile.d/conda.sh
conda activate qwen-cu128
export PYTHONNOUSERSITE=1

python demo_pipeline.py "$@"