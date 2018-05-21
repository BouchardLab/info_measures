#!/bin/bash -l
#SBATCH --qos=regular
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -J time_mi
#SBATCH -o time_mi.o%j


if [ "$NERSC_HOST" == "edison" ]
then
  cores=24
fi
if [ "$NERSC_HOST" == "cori" ]
then
  cores=32
fi

source activate info

echo $(which python)
echo $PATH

srun -N 1 -n 1 -c "$cores" python -u $HOME/info_measures/scripts/time_mi_n_jobs.py
