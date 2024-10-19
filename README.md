# Quantum Annealer
C++ script (and Python) for simulating a qauntum annealer coupled to a thermal bath. Used to investigate thermal effects in quantum computations.

### Requisits
Make sure you have the following libraries installed
	- Eigen 
	- boost 
	- Progressbar (optional)
		https://github.com/gipert/progressbar
By executing IncludesPaths.sh via "source IncludePaths.sh", you can conviniently include the paths
of the Egan and Boost libraries to the $CPLUS_INCLUDE_PATH variable. Make sure the paths in the script
are correct. 



### Overview 
- Cpp_QA.cxx: (Main) C++ code for Quantum Annealer. 
- QA.py - Python code for Quantum Annealer used in developement stages. Runs, but not fully updated.
- SLURM_JOB_QA.sh - SLURM script for running Cpp_QA.cxx on Cosmos (cluster).
- IncludePaths-sh - Contains paths to libraries
- Statistics - Folder containing data I gathered for the presentation. 
