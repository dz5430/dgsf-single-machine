#!/bin/bash
#SBATCH --job-name=Clean_Sens_0  # Job name
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=4 # cpu-cores per task
#SBATCH --mem-per-cpu=3G # memory per cpu-core
#SBATCH --time 24:00:00 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin # send email when job begins
#SBATCH --mail-type=end # send email when job ends
#SBATCH --mail-user=dz5430@princeton.edu

module purge
module load gurobi/10.0.1
module load anaconda3/2023.3

# Activate the virtual environment
source myenv/bin/activate

cd /home/dz5430

# Run the command for the job
python Cluster_solver_1_discrete_time.py

deactivate
