#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --account=usumae-np
#SBATCH --partition=usumae-np
#SBATCH -C c64
#SBATCH -o TO_Database-%j
#SBATCH --mail-user=christian.bolander@aggiemail.usu.edu
#SBATCH --mail-type=END

module purge
module load python/3.7.3

cd /uufs/chpc.utah.edu/common/home/u0764388/BIRE/F-16-Aero-Database/TODatabase
python f16MUX.py > output-$SLURM_JOB_ID-TO.out
