#include <iostream>
#include <string>
#include <cstring>
#include <bitset>
#include <cstdlib>
#include <time.h>
#include <fstream>
#include <iomanip>
#include<unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

// Non-standard libraries
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "eigen-3.4.0/unsupported/Eigen/CXX11/Tensor"
#include "boost_1_82_0/boost/dynamic_bitset.hpp"
#include "progressbar-master/include/progressbar.hpp"

using namespace std;
using namespace Eigen;
using namespace std::chrono;


// =============== GLOBALLY DEFINED INPUTS ================== //
// ========================================================== //

const int num_qbits = 2;	// Number of Qbits
const double T = 1;		// Temperature
const double lam_sq = 1;	// Coupling strength




// =================== USEFUL FUNCTIONS ====================== //
// =========================================================== //

size_t PickRandomWeightedElement(const std::vector<double>& weights) {
	// Picks a random index in [0, weights.size()-1] according to the
	// distribution described by the weights given by std::vector input.

	double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
	double random_num = (static_cast<double>(std::rand()) / RAND_MAX)  * total_weight;	

	double cumulative_sum = 0;
	for (std::size_t i=0; i<weights.size(); i++) {
		cumulative_sum += weights[i];
		if (random_num < cumulative_sum) {
			return i;
		}
	}
	return -1;   //Should never occur
}


double Round(const double value, const double precision) {
	// Rounds double to given precision.	
	return round(value/precision) * precision;
}


VectorXd RoundVector(const VectorXd vector, const double precision) {
	// Rounds Eigen::vectors (or Eigen::Arrays) of doubles to given precision.
	// For 3 decimals, set precision to 0.001.
	
	VectorXd rounded_vector(vector.size());
	for (int i=0; i<vector.size(); i++) {
		double element_value = vector(i);
		rounded_vector(i) = Round(element_value, precision);
	}
	return rounded_vector;
}


ArrayXXd PSDE(ArrayXXd measurements) {
	// PSDE = Probabilities, Standard Deviations and Errors.
	// Given probabilities from multiple MCWF simulations, function computes the
	// average probability to be in each state, and the associated standard deviations and errors.
	
	// INPUT:
	// ------
	// 	measurements = eigen::Array of size Nxdim, where dim = 2**num_qbits and 
	// 	N = number of measurments/simulations.
	
	// OUTPUT: 
	// -------
	// 	statistics = eigen::Array of size 3xdim. vec(0) = probabilities,
	// 	vec(1) = standard deviations and vec(2) = standard errors.
	
	double N = measurements.rows();
	int num_cols = measurements.cols();
	ArrayXXd statistics(3, num_cols); 

	for (int i=0; i<num_cols; i++) {
		ArrayXd col_vec = measurements.col(i);
		statistics(0,i) = col_vec.mean();	// Average probabilities
		statistics(1,i) = std::sqrt((1.0/N) * (col_vec - col_vec.mean()).square().sum());	//
		statistics(2,i) = statistics(1,i)/std::sqrt(N);
	}
	return statistics;
}

int NumberOfLines(const string filename) {
	// Finds the number of lines in file with path filename //
	
	ifstream f;
	f.open(filename);
	if (f.fail()) {
	       	cerr << "Error: Unable to open file\n";
       	}

	string line;
	int num_lines = -1;
	while (!f.eof()) { // while not at the end of file
		num_lines++;
		getline(f, line);
	}
	f.close();

	return num_lines;
}

ArrayXXd ReadFile(const string filename, const int num_rows, const int num_cols) {
	// Reads data file and returns data as a 2D Eigen::Array. Need to specify
	// number of rows and columns in data file. Use function NumberOfLines.
	
	ifstream f;
	f.open(filename);
	vector<vector<double>> data(num_rows, vector<double>(num_cols)); 

	// Read data from file and store in 2D std::vector
	for (int i = 0; i < num_rows; i++) {                      
		for (int j = 0; j < num_cols; j++) {
			if (!(f >> data[i][j])) {
				cerr << "Error: Unable to read data from the file\n";
			}
		}
	}
	f.close();

	// Convert 2D std::vector to Eigen::ArrayXXd.
    	ArrayXXd probs(num_rows, num_cols);  			
    	for (int i = 0; i < num_rows; i++) {
        	for (int j = 0; j < num_cols; j++) {
            			probs(i, j) = data[i][j];
        	}
    	}

	return probs;
}






// ======= HAMILTONIANS, EIGENVECTORS, EIGENVALUES ========= //
// ========================================================= //

Matrix<int, Dynamic, Dynamic> Hamiltonian_sz(const int n) {
	// Computes the 'z-part' of the Hamiltonian:
	// 	\sigma_z^1 + \sigma_z^2 + ... + \sigma_z^n,
	// where n is the number of Qbits.
	
	char bit_val1 = '1';
	int dim = pow(2,n);
	MatrixXi H_sz(dim,dim);
	H_sz.setZero();

	for (int index=0; index<dim; index++) {	
		boost::dynamic_bitset<> bin_index(n, index);
		string bin_index_string;
		to_string(bin_index, bin_index_string);
		
		int diag_val = 0;
		for (int bit=0; bit<n; bit++) {
			char bit_val = bin_index_string[bit];
			if (bit_val == bit_val1) {
				diag_val++;
			} else {
				diag_val--;
			}
		H_sz(index,index) = diag_val;
		}
	}
	return H_sz;
}


Matrix<int, Dynamic, Dynamic> Hamiltonian_sx(const int n) { 
	// Computes the 'x-part' of the Hamiltonian:
	// 	\sigma_x^1 + \sigma_x^2 + ... + \sigma_x^n,
	// where n is the number of Qbits.
	
	char bit_val1 = '1';
        int dim = pow(2,n);
        MatrixXi H_sx(dim,dim);
        H_sx.setZero();

        for (int index=0; index<dim; index++) {
		boost::dynamic_bitset<> bin_index(n, index);
		string bin_index_string;
		to_string(bin_index, bin_index_string);

                for (int bit=0; bit<n; bit++) {
			string bin_index_temp = bin_index_string;
                        char bit_val = bin_index_string[bit];
                        if (bit_val == bit_val1) {
                        	bin_index_temp[bit] = '0';
			} else {
                        	bin_index_temp[bit] = '1';
			}

			boost::dynamic_bitset<> b(bin_index_temp);
			int flipped_index = b.to_ulong();
			H_sx(index,flipped_index) = 1;
                }
        }
        return H_sx;
}


struct EigenSystem {
	// Structure containing Hamiltonian and corresponding eigenvectors
	// and eigenvalues as members.

		MatrixXd* H;
		MatrixXd* eigenstates;
		VectorXd* eigenvalues;
	};


EigenSystem Hamiltonian(const double s, const int n) {
	// Computes the Hamiltonian for n Qbits:
	// (\sigma_x^1 + ... + \sigma_x^n) * (1-s) + (\sigma_z^1 + ... + \sigma_z^n) * s,
	// where 0 =< s =< 1. Corresponding eigenvectors and eigenvalues also computed.
	// Returns EigenSystem data type.

	int dim = pow(2,n);

	EigenSystem eigensystem;
	eigensystem.H = new MatrixXd(dim,dim);
	eigensystem.eigenstates = new MatrixXd(dim,dim);
	eigensystem.eigenvalues = new VectorXd(dim);
	
	MatrixXi H_sz = Hamiltonian_sz(n);
	MatrixXi H_sx = Hamiltonian_sx(n);

	for (int i=0; i<dim; i++) {
		for (int j=0; j<dim; j++) {
			(*eigensystem.H)(i,j) = H_sx(i,j) * (1-s) + H_sz(i,j) * s;
		}
	}

	SelfAdjointEigenSolver<MatrixXd> eigensolver;
	eigensolver.compute(*eigensystem.H);
	*eigensystem.eigenstates = (eigensolver.eigenvectors()).transpose();
	*eigensystem.eigenvalues << eigensolver.eigenvalues();

	//cout << *eigensystem.eigenvectors << "\n";
	//cout << (*eigensystem.eigenvalues).transpose() << "\n";
	//cout << *eigensystem.H << "\n";

	return eigensystem;
}




// ============ MONTE CARLO WAVE-FUNCTION ============= //
// ==================================================== //

double N(const double x, const double y) {
	// Average number of photons in thermal bath with  energy x-y
	// where energy of first state is x, and energy of second state
	// is y. Temperature T (globally defined).
	double beta = 1/T;
	double N = 1/(exp(beta*(fabs(x-y)))-1);
	return N;
} 


VectorXcd C(const int n, const int a, const int b, const VectorXcd phi) {
	// C_m operator. Requires states a, b that are involved in quantum jump
	// and the wavefunction phi to act on, where 0 <= a,b < 2**num_qbits.

	int dim = pow(2,n);
	VectorXcd phi_prime(dim);
	phi_prime.setZero();
	phi_prime(b) = phi(a);
	return phi_prime;
}


bool MCWF(VectorXcd phi, const int n, const bool save_phi_history, VectorXd& probs) {
	// Performs quantum annealing using the Monte-Carlo Wave Function based on Mølmer.
	// INPUT:
	// ------
	// VectorXcd phi = 	Initial state vector of size 2**num_qbits. Commonly 
	// 			chosen to be the groundstate of the initial Hamiltonian.
	// int n =		Number of Qbits.
	// bool save_phi_history = if true, will save phi from each iteration.
	// VectorXd& probs = 	Empty declared vector of size 2**num_qbits. Variable
	// 			will be updated by the MCWF if method is successfull.
	// 			Contains the probabilities of finding the system in
	// 			each of the (classical) states.
	// OUTPUT:
	// ------	
	// bool, true if MCWF was successfull, false otherwise. If true, input parameter
	// probs is updated accordingly.
	// Additional parameters required below.


	// Additional parameters
	const double dt = 0.0025;	// Free parameter: Step size.
	const double AT = 13; 		// Free parameter: Anneal time
					
	const int iterations = static_cast<int>(AT/dt);
	int dim = pow(2,n);

	//cout << "\nInitial phi \n" << phi.transpose() << "\n\n";

	// Save wavefunction from each iteration expressed in z and 
	// instataneous (i) basis, respectively.
	ArrayXXd phi_history_z(iterations+1, dim);
	ArrayXXd phi_history_i(iterations+1, dim);

	VectorXcd phi_i(dim); 	// phi in instataneous basis

	//progressbar bar(iterations+1);		// Uncomment to view progressbar (1/2)
	//auto start = high_resolution_clock::now();	// Uncomment to measure time (1/4)
	for (int i=0; i<=iterations; i++) {
		//bar.update();				// Uncomment to view progressbar (2/2)
		EigenSystem eigensystem = Hamiltonian(i*(dt/AT), n);
		phi_i = (*eigensystem.eigenstates) * phi.conjugate();	// z -> instataneous basis


		std::vector<double> pre_factors;
		int photon_type; 			// Type of quantum jump. 1 for abs, -1 for em, 0 for none.
		std::vector<double> counter_list1;	// Elements are a,b for the different delta_p_m,
							// where the quantum jump is between (classical) states a -> b.
		std::vector <int> counter_list2;	// Elements are type of quantum jump (photon_type) for given delta_p_m
	

		double delta_p = 0;
		std::vector<double> delta_p_list;	// Elements are first order delta_p_m values
			
		// System goes from state a -> b during emission/absorption	
		for (int a=0; a<dim; a++) {
			double energy_a = (*eigensystem.eigenvalues)(a);
			for (int b=0; b<dim; b++) {
				double energy_b = (*eigensystem.eigenvalues)(b);
				if (a==b) { 
					continue;
				}
				// Spontaneous emission (energy_a > energy_b)
				if (Round(energy_a-energy_b, 0.01) > 0) {
					pre_factors.push_back((N(energy_a, energy_b)+1)*lam_sq);
					photon_type = -1;
				// Absorption (energy_a < energy_b)
				} else if (Round(energy_b-energy_a, 0.01) > 0) {
					pre_factors.push_back(N(energy_b, energy_a)*lam_sq);
					photon_type = 1;
				// Degenerate eigenvalues (energy_a = energy_b)
				} else {
					pre_factors.push_back(0); 
					photon_type = 0;	
				}
				counter_list1.push_back(a);
				counter_list1.push_back(b);
				counter_list2.push_back(photon_type);
				complex<double> temp1 = phi_i.adjoint() * C(n,b,a, C(n,a,b,phi_i));		//Can be made faster!
				double temp2 = temp1.real();
				delta_p_list.push_back(pre_factors.back() * temp2 * dt);


				// delta_p second order (1/2)
				for (int c=0; c<dim; c++) {
                                	double energy_c = (*eigensystem.eigenvalues)(c);
                                        double pre_factor2;	// pre_factor for second order terms of delta_p
                                        if (Round(energy_a-energy_c, 0.01) > 0) {
                                                pre_factor2 = (N(energy_a, energy_c)+1)*lam_sq;
                                        } else if (Round(energy_c-energy_a, 0.01) > 0) {
                                                pre_factor2 = N(energy_c, energy_a)*lam_sq;
                                        } else {
                                                pre_factor2 = 0;

					delta_p -= (1.0/4.0) * pre_factor2 * pre_factors.back() * temp2 * pow(dt,2);
                                        }
				}
			}
		}
		// delta_p first order
		delta_p += std::accumulate(delta_p_list.begin(), delta_p_list.end(), 0.0);
		// delta_p second order (2/2)
		complex<double> temp3 = phi_i.adjoint() * (*eigensystem.H * *eigensystem.H) * phi_i;
		delta_p -= temp3.real() * pow(dt,2);

		if (delta_p > 0.1) {
			cout << "\nWarning! delta_p  is getting large, must be much smaller than 1. Current value:" << delta_p << "\n";
		}

			
		double epsilon = (static_cast<double>(std::rand()) / RAND_MAX);	// Random number in [0,1]
		VectorXcd phi_new(dim);
		// No emission/absorption
		if (epsilon > delta_p) {
			VectorXcd phi_1 = phi_i - complex<double>(0,1) * (*eigensystem.H * phi_i) * dt;
			int counter = 0;
			for (int a=0; a<dim; a++) {
				double energy_a = (*eigensystem.eigenvalues)(a);
			for (int b=0; b<dim; b++) {
				double energy_b = (*eigensystem.eigenvalues)(b);
				if (a==b) {
					continue;
					}
					phi_1 -= 0.5 * pre_factors[counter] * C(n,b,a, C(n,a,b,phi_i)) * dt;
					counter++;
				}
			}
			phi_new = phi_1/pow(1.0-delta_p,0.5);
		// Emission/absorption
		} else {
			// Pick delta_p_m and unpack information from counter_lists
			// for given m.
			size_t index =  PickRandomWeightedElement(delta_p_list);	
			double delta_p_m = delta_p_list[index];
			int a = counter_list1[2*index];
			int b = counter_list1[2*index+1];
			int photon_type = counter_list2[index];

			// Uncomment below to print when quantum jumps occur. Only use
			// while running a single MCWF simulation.
			//if (photon_type==-1) {
			//	cout << "\nSpontaneous emission\n";
			//} else if (photon_type==1) {
			//	cout << "\nAbsorption!\n";
			//} else {
			//	cout << "\nODD!\n";
			//}

			phi_new = pow(pre_factors[index], 0.5) * (C(n,a,b,phi_i)/pow(delta_p_m/dt,0.5)); 		
		}

		
		// Instataneous -> z-basis
		phi = (*eigensystem.eigenstates).transpose() * phi_new;

		
		ArrayXd probs_z = phi.normalized().array().abs2();
		ArrayXd probs_i = phi_i.normalized().array().abs2();
		phi_history_z.row(i) = RoundVector(probs_z, 0.001);
		phi_history_i.row(i) = RoundVector(probs_i, 0.001);


		delete eigensystem.H;
        	delete eigensystem.eigenstates;
        	delete eigensystem.eigenvalues;

		// If wavefunction norm is deviating from 1, return false and stop simulation.
		double phi_norm = phi.norm();
		if (phi_norm>1.1 or phi_norm<0.9) {
			cout << "#######################################################################################\n";
			cout << "Phi was not normalized properly, aborting at  " << static_cast<double>(i)/iterations << "\n";
			cout << "phi norm: " << phi_norm << "\n\n";
			return false;
		}

	}
	//auto stop = high_resolution_clock::now(); 			// Uncomment to measure time (2/4)
	//auto duration = duration_cast<milliseconds>(stop - start); 	// Uncomment to measure time (3/4)
								   
	if (save_phi_history) {
		ofstream f;
		f.open("PhiHistoryCpp_i.txt", ios::out);
		f << phi_history_i;
		f.close();
		f.open("PhiHistoryCpp_z.txt", ios::out);
		f << phi_history_z;
		f.close();
	}

	
	VectorXd probs_unormalized = phi_i.array().abs2();
	probs =  phi_i.normalized().array().abs2();
	//cout << "\n\n------------- SUMMARY ------------\n";
	//cout << "Probabilities\n" << RoundVector(probs_unormalized, 0.001).transpose() << "\n";
	//cout << "Phi norm: " << Round(phi_i.norm(), 0.001) << "\n";
	//cout << "Normalized probabilities\n" << RoundVector(probs, 0.001).transpose() << "\n";
	//cout << "Duration: " << duration.count() << endl;	// Uncomment to measure time (4/4)	
	//cout << "---------------- END ---------------\n";

	return true;
}


void RunMCWF(VectorXcd phi, const int n) {
	// Runs MCWF using multiprocessing. Additional input required below.

	// Additional input
	string filename = "AverageProbCpp" + to_string(num_qbits) + "Q.txt";	// Output file
	int num_proc = 5;		// Number of simulations to run simultaneously
	int num_simulations = 100;	// Total number of simulations to run. 
					// Pick num_proc/num_simulations = int number

	cout << "Initial_phi\n" << phi.transpose() << "\n";

	// Use multiprocessing to run MCWF multiple times.
	// Save results from each run in text file.
	ofstream f1;
	f1.open(filename, ios::out);
	pid_t pid, wpid;
	int iterations = num_simulations/num_proc;
	for (int i=0; i<num_proc; i++) {
		pid = fork();
		if (pid==0) {	// If child, perform MCWF
			// Generate new seed for the random number generator
			// for each child process. Use PID as the seed.
			unsigned int seed = static_cast<unsigned int>(getpid());
			srand(seed);

			// Each child process runs MCWF multiple times.
			for (int j=0; j<iterations; j++) {
				VectorXd probs;
				if (!MCWF(phi, num_qbits, false, probs)) {
					continue;
				}	
				f1 << probs.transpose() << "\n";
			}
			f1.close();
			_exit(0);	// Terminate child process

		} else if (pid<0) {
			cerr << "Fork failed!\n";
			return;
		}	
	}	
	f1.close();


	// Parent process waits for all children to terminate.	
	int status = 0;
	while ((wpid = wait(&status)) > 0);

	// Read data from file and give a summary.
	int num_rows = NumberOfLines(filename);
        int num_cols = pow(2, n);
	ArrayXXd probs = ReadFile(filename, num_rows, num_cols);

	// Call PSDE function to compute average probability, standard
	// deviation and error for being in each state.
	ArrayXXd statistics(3,probs.cols());
	statistics = PSDE(probs);


	cout << "\n\n------------------- SUMMARY ---------------------\n";
	cout << "Average Probabilities: " << statistics.row(0) << "\n";
	cout << "Standard Deviations:   " << statistics.row(1) << "\n";
	cout << "Standard Errors:       " << statistics.row(2) << "\n\n\n";
	cout << "\n\n-------------------- END ----------------------\n";
	
	return;	
}


void RunMCWF_Lunarc(VectorXcd phi, const int n) {
	// Runs MCWF, to be used with SLURM jobs on Lunarc. If you want to run a 
	// large number of simulations on Lunarc, and don't want to sleep between each
	// job, set num_simulations to int greater than 1. This will run MCWF several times
	// for each job on Lunarc. 
	// Additional input required below.

	// Additional input
	int num_simulations = 1;
	string filename = "AverageProbCpp" + to_string(num_qbits) + "Q.txt";

	// Use multiprocessing to run MCWF multiple times.
	// Save results from each run in text file.
	ofstream f1;
	f1.open(filename, ios::out | ios::app);

	// Generate new seed for the random number generator
	unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
	srand(seed);

	for (int i=0; i<num_simulations; i++) {
		VectorXd probs;
		if (!MCWF(phi, num_qbits, false, probs)) {
			continue;
		}
			cout << "DONE\n";	
			f1 << probs.transpose() << "\n";
	}
	f1.close();

	return;	
}




// ======= BOLTZMAN AND OTHER USEFUL FUNCTIONS ======== //
// ==================================================== //


double BoltzmanDist(const int n) {
	// Given temperature T (globally defined) and number of Qbits, 
	// function computes the probability of being in the groundstate
	// when in final Hamiltonian (H_z) according to Boltzman.
	
	EigenSystem eigensystem = Hamiltonian(1, n);
	double eigenvalue_min = (*eigensystem.eigenvalues).minCoeff();
	
	double Z = 0;
	for (int i=0; i<eigensystem.eigenvalues->size(); i++) {
		Z += exp(-(*eigensystem.eigenvalues)(i)/T);
	}
	double boltzman_prob = exp(-eigenvalue_min/T)/Z;

	delete eigensystem.H;
        delete eigensystem.eigenstates;
        delete eigensystem.eigenvalues;

	return Round(boltzman_prob, 0.001);
}

void CompareBoltzmanToMCWF(const int n) {
	// This function is used when MCWF has been used to gather data for
	// different sizes of systems. The function will read this data from
	// string filename and create a new file containing the following information:
	// - Probability for being in the groundstate according to MCWF
	// - Standard deviation
	// - Standard error
	// - Same probability according to Boltzman
	// It will do the above for 2,3,..,n Qbits (one Qbit in each row) 
	// Make sure that filename below matches where you have stored you data
	// from the MCWF simulations!


	std::vector<double> plot_probs;
	std::vector<double> plot_std_devs;
	std::vector<double> plot_std_errors;

	for (int i=2; i<=n; i++) {
		string filename = "Statistics/AverageProbCpp" + to_string(i) + "Q.txt";

		int num_rows = NumberOfLines(filename);
		int num_cols = pow(2, i); 
		ArrayXXd probs = ReadFile(filename, num_rows, num_cols);


		ArrayXXd statistics(3,probs.cols());
		statistics = PSDE(probs);

		plot_probs.push_back(statistics(0,0));
		plot_std_devs.push_back(statistics(1,0));
		plot_std_errors.push_back(statistics(2,0));
	}

	// Save to new file
	ofstream f;
	f.open("Statistics/AverageProbCppFinal.txt");
	for (int i=0; i<plot_probs.size(); i++) {
		double boltzman_prob = BoltzmanDist(i+2);
		cout << boltzman_prob << endl;
		f << i+2 << std::setw(10) << Round(plot_probs[i],0.0001) << std::setw(10) << Round(plot_std_devs[i],0.0001) << std::setw(10) <<  Round(plot_std_errors[i],0.0001) << std::setw(10) << Round(boltzman_prob,0.0001) << "\n";
	}
	f.close();
}


void ComputeAverageProb(const int n) {
	// Computes the average probability of being in the groundstate by reading
	// data from file (created by MCWF). Give correct size of system (n) and make
	// sure filename below is correct.
        string filename = "AverageProbCpp" + to_string(num_qbits) + "Q.txt";

        // Read data from file and give a summary.
        int num_rows = NumberOfLines(filename);
        int num_cols = pow(2, n);
        ArrayXXd probs = ReadFile(filename, num_rows, num_cols);

        // Call PSDE function to compute average probability, standard
        // deviation and error for being in each state.
        ArrayXXd statistics(3,probs.cols());
        statistics = PSDE(probs);


	double b_prob = BoltzmanDist(num_qbits);

        cout << "\n\n------------------- SUMMARY ---------------------\n";
        cout << "Average Probabilities: " << statistics.row(0) << "\n";
        cout << "Standard Deviations:   " << statistics.row(1) << "\n";
        cout << "Standard Errors:       " << statistics.row(2) << "\n\n\n";
        cout << "Boltzman Distribution: " << b_prob << "\n";
        cout << "----------------------- END ------------------------\n";

	return;
}




int main() {
	int dim = pow(2,num_qbits);

	// Initial wave function = groundstate of initial Hamiltonian
	EigenSystem eigensystem = Hamiltonian(0, num_qbits);
	VectorXd initial_phi(dim);
	initial_phi = eigensystem.eigenstates->row(0);

	// To run the MCWF directly, uncomment below.
	//unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
        //srand(seed);
	//VectorXd probs;
	//MCWF(initial_phi, num_qbits, true, probs);

	
	//RunMCWF(initial_phi, num_qbits);
	//RunMCWF_Lunarc(initial_phi, num_qbits);
	//TempFcn(num_qbits);
	
	//cout << "Boltzman: " << BoltzmanDist(6) << endl;
	
	//CompareBoltzmanToMCWF(9);

	delete eigensystem.H;
        delete eigensystem.eigenstates;
        delete eigensystem.eigenvalues;
	return 0;
}













