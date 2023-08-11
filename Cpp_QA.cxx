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

const int num_qbits = 7;
//const int dim = pow(2,num_qbits);
const double T = 1;	//0.5/1
const double lam_sq = 1;	//1




// =================== USEFUL FUNCTIONS ====================== //
// =========================================================== //

size_t PickRandomWeightedElement(const std::vector<double>& weights) {
	// Picks a random index [0, weights.size()-1] according to the
	// distribution described by the weights std::vector.

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
	// Rounds eigen vectors of doubles to given precision.
	
	VectorXd rounded_vector(vector.size());
	for (int i=0; i<vector.size(); i++) {
		double element_value = vector(i);
		rounded_vector(i) = Round(element_value, precision);
	}
	return rounded_vector;
}


ArrayXXd PSDE(ArrayXXd measurements) {
	// PSDE = Probabilities, Standard Deviations and Errors.
	// Computes the probability to be in each state, and the associated
	// standard deviations and errors.
	
	// INPUT:
	// ------
	// 	measurements = eigen::Array of size Nxdim, where dim = 2**n and 
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
		statistics(0,i) = col_vec.mean();
		statistics(1,i) = std::sqrt((1.0/N) * (col_vec - col_vec.mean()).square().sum());
		statistics(2,i) = statistics(1,i)/std::sqrt(N);
	}
	return statistics;
}

int NumberOfLines(const string filename) {
	// Finds the number of lines in file with path file_path //
	
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
	// Reads file and returns data as a 2D Eigen::Array. Need to specify
	// number of rows and columns in data file.
	
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
	// where n are the number of qbits.
	
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
	// where n are the number of qbits.
	
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
	// Structure with Hamiltonian and corresponding eigenvectors
	// and eigenvalues as members.

		MatrixXd* H;
		MatrixXd* eigenstates;
		VectorXd* eigenvalues;
	};


EigenSystem Hamiltonian(const double s, const int n) {
	// Computes the Hamiltonian for n qbits:
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
	double beta = 1/T;
	double N = 1/(exp(beta*(fabs(x-y)))-1);
	return N;
} 


VectorXcd C(const int n, const int a, const int b, const VectorXcd phi) {
	int dim = pow(2,n);
	VectorXcd phi_prime(dim);
	phi_prime.setZero();
	phi_prime(b) = phi(a);
	return phi_prime;
}


bool MCWF(VectorXcd phi, const int n, const bool print_summary, VectorXd& probs) {
	//cout << "\nInitial phi \n" << phi.transpose() << "\n\n";
	int dim = pow(2,n);

	const double dt_ds_ratio = 0.9; // 1.8
	const double ds = 0.000005;	//0.0001
	const double dt = dt_ds_ratio * ds;
	const int iterations_s = static_cast<int>(1/ds);
	const int iterations_t = 10;	//10/3

	ArrayXXd phi_history_z(iterations_s+1, dim);
	ArrayXXd phi_history_x(iterations_s+1, dim);

	VectorXcd phi_decomp(dim);

	//progressbar bar(iterations_s+1);
	auto start = high_resolution_clock::now();
	for (int i=0; i<=iterations_s; i++) {
		//bar.update();
		double s = i * ds;
		EigenSystem eigensystem = Hamiltonian(s, n);
		// z-basis to instataneous basis
		//phi_decomp = (*eigensystem.eigenstates).conjugate() * phi;
		phi_decomp = (*eigensystem.eigenstates) * phi.conjugate();

		for (int j=0; j<iterations_t; j++) {
			std::vector<double> pre_factors;
			std::vector<double> counter_list1;
			std::vector <int> counter_list2;
			std::vector<double> delta_p_list;
			int photon_type; 		// 1 for abs, -1 for em, 0 for none
			
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
					complex<double> temp1 = phi_decomp.adjoint() * C(n,b,a, C(n,a,b,phi_decomp));		//Can be made faster!
					double temp2 = temp1.real();
					delta_p_list.push_back(pre_factors.back() * temp2 * dt);
				}
			}
			double delta_p = std::accumulate(delta_p_list.begin(), delta_p_list.end(), 0.0);
			double epsilon = (static_cast<double>(std::rand()) / RAND_MAX);
			
			if (delta_p > 0.1) {
				cout << "\nWarning! delta_p  is getting large, must be much smaller than 1. Current value:" << delta_p << "\n";
			}

			
			VectorXcd phi_new(dim);
			// No emission/absorption
			if (epsilon > delta_p) {
			//if (true) {
				VectorXcd phi_1 = phi_decomp - complex<double>(0,1) * (*eigensystem.H * phi_decomp) * dt;
				int counter = 0;
				for (int a=0; a<dim; a++) {
					double energy_a = (*eigensystem.eigenvalues)(a);
					for (int b=0; b<dim; b++) {
						double energy_b = (*eigensystem.eigenvalues)(b);
						if (a==b) {
							continue;
						}
						phi_1 -= 0.5 * pre_factors[counter] * C(n,b,a, C(n,a,b,phi_decomp)) * dt;
						counter++;
					}
				}
				phi_new = phi_1/pow(1.0-delta_p,0.5);
			// Emission/absorption
			} else {
				size_t index =  PickRandomWeightedElement(delta_p_list);	
				double delta_p_m = delta_p_list[index];
				int a = counter_list1[2*index];
			       	int b = counter_list1[2*index+1];
				int photon_type = counter_list2[index];

				//if (photon_type==-1) {
				//	cout << "\nSpontaneous emission\n";
				//} else if (photon_type==1) {
				//	cout << "\nAbsorption!\n";
				//} else {
				//	cout << "\nODD!\n";
				//}

				phi_new = pow(pre_factors[index], 0.5) * (C(n,a,b,phi_decomp)/pow(delta_p_m/dt,0.5)); 		
			}
			phi_decomp = phi_new;


		}
		
		// Instataneous basis to z-basis
		phi = (*eigensystem.eigenstates).transpose() * phi_decomp;

		
		ArrayXd probs_z = phi.normalized().array().abs2();
		ArrayXd probs_x = phi_decomp.normalized().array().abs2();
		phi_history_z.row(i) = RoundVector(probs_z, 0.001);
		phi_history_x.row(i) = RoundVector(probs_x, 0.001);


		delete eigensystem.H;
        	delete eigensystem.eigenstates;
        	delete eigensystem.eigenvalues;

		// If phi is not normalized properly, return an empty phi.
		double phi_len = phi.norm();
		if (phi_len>1.1 or phi_len<0.9) {
			cout << "#######################################################################################\n";
			cout << "Phi was not normalized properly, aborting at  " << static_cast<double>(i)/iterations_s << "\n";
			cout << "phi norm: " << phi_len << "\n\n";
			return false;
		}

	}
	auto stop = high_resolution_clock::now();

	if (print_summary) {
		ofstream f;
		f.open("PhiHistoryCpp_x.txt", ios::out);
		f << phi_history_x;
		f.close();
		f.open("PhiHistoryCpp_z.txt", ios::out);
		f << phi_history_z;
		f.close();
	}

	
	VectorXd probs_unormalized = phi_decomp.array().abs2();
	probs =  phi_decomp.normalized().array().abs2();
	auto duration = duration_cast<milliseconds>(stop - start);
	//cout << "\n\n------------- SUMMARY ------------\n";
	//cout << "Probabilities\n" << RoundVector(probs_unormalized, 0.001).transpose() << "\n";
	cout << "Phi norm: " << Round(phi_decomp.norm(), 0.001) << "\n";
	//cout << "Normalized probabilities\n" << RoundVector(probs, 0.001).transpose() << "\n";
	//cout << "Duration: " << duration.count() << endl;
	//cout << "---------------- END ---------------\n";

	return true;
}


void RunMCWF(VectorXcd phi, const int n) {
	// Runs MCWF using multiprocessing. Additional input required below.

	// Additional input
	int num_proc = 33;
	int num_simulations = 99;
	int iterations = num_simulations/num_proc;
	string filename = "AverageProbCpp" + to_string(num_qbits) + "Q.txt";

	cout << "Initial_phi\n" << phi.transpose() << "\n";

	// Use multiprocessing to run MCWF multiple times.
	// Save results from each run in text file.
	ofstream f1;
	f1.open(filename, ios::out);
	pid_t pid, wpid;
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
	
	return;	
}

double BoltzmanDist(const int n) {
	EigenSystem eigensystem = Hamiltonian(0, n);
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

	// Save ...
	ofstream f;
	f.open("Statistics/AverageProbCppFinal.txt");
	for (int i=0; i<plot_probs.size(); i++) {
		double boltzman_prob = BoltzmanDist(i+2);
		cout << boltzman_prob << endl;
		f << i+2 << std::setw(10) << plot_probs[i] << std::setw(10) << plot_std_devs[i] << std::setw(10) <<  plot_std_errors[i] << std::setw(10) << boltzman_prob << "\n";
	}
	f.close();
}


int main() {
	//srand(time(0));
	int dim = pow(2,num_qbits);
	EigenSystem eigensystem = Hamiltonian(0, num_qbits);
	VectorXd initial_phi(dim);
	initial_phi = eigensystem.eigenstates->row(0);
	VectorXd probs;

	//MCWF(initial_phi, num_qbits, true, probs);
	RunMCWF(initial_phi, num_qbits);
	double b_prob = BoltzmanDist(num_qbits);
	cout << "Boltzman Distribution: " << b_prob << "\n";
	
	//CompareBoltzmanToMCWF(5);

	delete eigensystem.H;
        delete eigensystem.eigenstates;
        delete eigensystem.eigenvalues;
	return 0;
}













