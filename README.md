### BEFORE RUNNING
-----------------------------------------------
Before running, make sure you have the following libraries installed
	- Eigen 
	- boost 
	- Progressbar (optional)
		https://github.com/gipert/progressbar
By executing IncludesPaths.sh via "source IncludePaths.sh", you can conviniently include the paths
of the Egan and Boost libraries to the $CPLUS_INCLUDE_PATH variable. Make sure the paths in the script
are correct. 



### FILES 
- Cpp_QA.cxx: (Main) C++ code for Quantum Annealer. 
- QA.py - Python code for Quantum Annealer used in developement stages. Runs, but not fully updated.
- SLURM_JOB_QA.sh - SLURM script for running Cpp_QA.cxx on Cosmos (cluster).
- IncludePaths-sh - Contains paths to libraries
- Statistics - Folder containing data I gathered for the presentation. 
