#!/bin/bash
#SBATCH --job-name="JSE_Trend_Pipeline_TCN_T_1K"
#SBATCH --account=compsci
#SBATCH --partition=swan
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=jse_pipeline_%j.out
#SBATCH --mail-user=mkhsiy057@myuct.ac.za
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

echo "====================================================="
echo "Job started on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Nodes: $SLURM_JOB_NODELIST"
echo "SLURM CPUs per Task: $SLURM_CPUS_PER_TASK" # This will now be '8'
echo "====================================================="

module load python/miniconda3-py3.12

echo "Running the main script using Python from 'myenv'..."
# We no longer need to pass the core count as an argument
myenv/bin/python main.py

echo "====================================================="
echo "Job finished at $(date)"
echo "====================================================="