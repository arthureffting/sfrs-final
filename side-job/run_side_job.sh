#!/bin/bash
#SBATCH --job-name=MY_EXAMPLE_JOB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -o /cluster/%u/sfrs/slurm/results/%j.out
#SBATCH -e /cluster/%u/sfrs/slurm/results/%j.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds"
#SBATCH --time=24:00:00

# Tell pipenv to install the virtualenvs in the cluster folder

export WORKON_HOME==/cluster/$(whoami)/.python_cache
export PYTHONPATH=.

# Small Python packages can be installed in own home directory. Not recommended for big packages like tensorflow -> Follow instructions for pipenv below
# cluster_requirements.txt is a text file listing the required pip packages (one package per line)

pip3 install --user -r requirements.txt

python3 side-job/extract_lines.py --base_folder=/cluster/$(whoami)/sfrs
