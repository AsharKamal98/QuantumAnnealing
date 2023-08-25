#!/bin/sh

#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=5300


#SBATCH -J "6NNE"
#SBATCH -o Ashar_%j.out
#SBATCH -e Ashar_%j.out


#SBATCH --time=03:00:00
#SBATCH -A lu2023-2-9 


# Create a directory where the output from Workscript.sh (MCWF) is re-directed.
mkdir -p OutputDir
rm -r OutputDir/*

export num_of_jobs=100
for ((i=0; i<num_of_jobs; i++))
do
	srun -Q -n 1 -N 1 --cpus-per-task=1 Workscript.sh &> "OutputDir/output${i}.txt" & 
	#Workscript.sh executes a.out

	# Code generates seed according to time. Wait 1-2 seconds between each job
	# to ensure that the seed is different.
	sleep 3
done

#wait for background processes to finish
wait

