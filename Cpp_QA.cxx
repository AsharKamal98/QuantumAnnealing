#include <iostream>
#include <string>
#include <cstring>
#include <bitset>
#include <cstdlib>
#include <time.h>
#include <fstream>
#include <iomanip>

// Non-standard libraries
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "eigen-3.4.0/unsupported/Eigen/CXX11/Tensor"
#include "boost_1_82_0/boost/dynamic_bitset.hpp"
#include "progressbar-master/include/progressbar.hpp"

using namespace std;
using namespace Eigen;


// =============== GLOBALLY DEFINED INPUTS ================== //
// ========================================================== //

const int num_qbits = 2;
const int dim = pow(2,num_qbits);
const double T = 0.1;
const double lam_sq = 1;




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


ArrayXXd Std_Dev_Err(ArrayXXd vec) {
	double N = vec.rows();
	ArrayXXd std_dev_err(3,vec.cols());

	for (int i=0; i<dim; i++) {
		ArrayXd col_vec = vec.col(i);
		std_dev_err(0,i) = col_vec.mean();
		std_dev_err(1,i) = std::sqrt((1.0/N) * (col_vec - col_vec.mean()).square().sum());
		std_dev_err(2,i) = std_dev_err(1,i)/std::sqrt(N);
	}
	return std_dev_err;
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
	//cout << *eigensystem.eigenvalues << "\n";
	//cout << *eigensystem.H << "\n";

	return eigensystem;
}




// ============ MONTE CARLO WAVE-FUNCTION ============= //
// ==================================================== //

double N(const double x, const double y) {
	double beta = 1/T;
	double N = 1/(exp(beta*(abs(x-y)))-1);
	return N;
} 


VectorXcd C(const int a, const int b, const VectorXcd phi) {
	VectorXcd phi_prime(dim);
	phi_prime.setZero();
	phi_prime(b) = phi(a);
	return phi_prime;
}


ArrayXd MCWF(VectorXcd phi, const int n) {
	cout << "\nInitial phi \n" << phi.transpose() << "\n\n";

	const double dt_ds_ratio = 0.5;
	const double ds = 0.020;
	const double dt = dt_ds_ratio * ds;
	const int iterations_s = static_cast<int>(1/ds);
	const int iterations_t = 1;

	ArrayXXd phi_history_z(iterations_s+1, dim);
	ArrayXXd phi_history_x(iterations_s+1, dim);

	VectorXcd phi_decomp(dim);
	VectorXd eigenval_temp(dim);

	cout << "Number of iterations_s: " << iterations_s << "\n";
	progressbar bar(iterations_s+1);
	for (int i=0; i<=iterations_s; i++) {
		bar.update();
		double s = i * ds;
		EigenSystem eigensystem = Hamiltonian(s, n);
		phi_decomp = (*eigensystem.eigenstates).conjugate() * phi;

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
						pre_factors.push_back(1);
						photon_type = 0;	
					}
					counter_list1.push_back(a);
					counter_list1.push_back(b);
					counter_list2.push_back(photon_type);
					complex<double> temp1 = phi_decomp.conjugate().transpose() * C(b,a, C(a,b,phi_decomp));
					double temp2 = temp1.real();
					delta_p_list.push_back(pre_factors.back() * temp2 * dt);


				}
			}
			double delta_p = std::accumulate(delta_p_list.begin(), delta_p_list.end(), 0.0);
			double epsilon = (static_cast<double>(std::rand()) / RAND_MAX);
			
			if (delta_p > 0.1) {
				cout << "Warning! delta_p  is getting large, must be much smaller than 1. Current value:" << delta_p << "\n";
			}

			
			VectorXcd phi_new(dim);
			// No emission/absorption
			if (epsilon > delta_p) {
				VectorXcd phi_1 = phi_decomp - complex<double>(0,1) * (*eigensystem.H * phi_decomp) * dt;
				int counter = 0;
				for (int a=0; a<dim; a++) {
					double energy_a = (*eigensystem.eigenvalues)(a);
					for (int b=0; b<dim; b++) {
						double energy_b = (*eigensystem.eigenvalues)(b);
						if (a==b) {
							continue;
						}
						phi_1 -= 0.5 * pre_factors[counter] * C(b,a, C(a,b,phi_decomp)) * dt;
						counter++;

					}
				}
				phi_new = phi_1/pow(1-delta_p,0.5);
			// Emission/absorption
			} else {
				size_t index =  PickRandomWeightedElement(delta_p_list);	
				double delta_p_m = delta_p_list[index];
				int a = counter_list1[2*index];
			       	int b = counter_list1[2*index+1];
				int photon_type = counter_list2[index];
				//cout << "\n";
				//cout << "Index: " << index << "\n";
				//cout << "delta_p_list ";
				//cout << "counter_list ";
				//for (int k=0; k<counter_list1.size(); k++) {
				//	cout << counter_list1[k] << " ";
				//}
				//cout << "\n";

				if (photon_type==-1) {
					cout << "\nSpontaneous emission\n";
					//cout << "Iteration :" << i << "\n";
				} else if (photon_type==1) {
					cout << "\nAbsorption!\n";
					//cout << "Iteration :" << i << "\n";
				} else {
					cout << "ODD!\n";
				}

				//cout << "Transition: " << a << b << "\n";
				//cout << "probabilities before" << phi_decomp.array().abs2().transpose() << "\n";
				phi_new = pow(pre_factors[index], 0.5) * (C(a,b,phi_decomp)/pow(delta_p_m/dt,0.5)); 		
				//cout << "probabilities after " << phi_new.array().abs2().transpose() << "\n\n";
			}
			phi_decomp = phi_new;


		}
		
		// Back to old basis
		phi = (*eigensystem.eigenstates).transpose() * phi_decomp;

		
		ArrayXd probs_z = phi.array().abs2();
		ArrayXd probs_x = phi_decomp.array().abs2();

		phi_history_z.row(i) = probs_z; //.transpose();
		phi_history_x.row(i) = RoundVector(probs_x, 0.001); //.transpose();

		//cout << "\neigenvalues: " << (*eigensystem.eigenvalues).transpose() << "\n";
		//cout << "eigenstates\n" << *eigensystem.eigenstates << "\n\n";

		delete eigensystem.H;
        	delete eigensystem.eigenstates;
        	delete eigensystem.eigenvalues;

	}


	//ArrayXd probs = phi.array().abs2();

	//ofstream f;
	//f.open("mydata.txt", ios::out);
	//for (int i=0; i<=iterations_s; i++) {
	//	f << phi_history_x(i,0) << std::setw(10) << phi_history_x(i,1) << std::setw(10) << phi_history_x(i,2) << std::setw(10) << phi_history_x(i,3) << "\n";
	//}
	//f << phi_history_x;
	//f.close();

	
	ArrayXd probs = phi.array().abs2();
	cout << "\n\n------------- SUMMARY ------------\n";
	//cout << "phi_decomp probabilities\n" << RoundVector(phi_decomp.array().abs2(), 0.01).transpose() << "\n";
	cout << "Probabilities\n" << probs.transpose() << "\n";
	cout << "Norm: " << Round(phi_decomp.norm(), 0.01) << "\n\n";

	return probs;
}


void BoltzmanCheck(VectorXcd phi, const int n) {
	int iterations = 3;
	string filename = "AverageProbCpp.txt";

	ArrayXXd probs(iterations, dim);
	//for (int i=0; i<iterations; i++) {
	//	probs.row(i) = MCWF(phi, num_qbits);	
	//}

	//ofstream f;
	//f.open(filename, ios::out);
	//f << probs;
	//f.close();

	probs << 1,2,3,4, 1,2,3,4, 2,4,6,8;
	ArrayXXd statistics(3,probs.cols());
	statistics = Std_Dev_Err(probs);

	cout << "\n\n------------------- SUMMARY ---------------------\n";
	cout << "Average Probabilities: " << statistics.row(0) << "\n";
	cout << "Standard Deviations: " << statistics.row(1) << "\n";
	cout << "Standard Errors: " << statistics.row(2) << "\n\n\n";
}


int main() {
	srand(time(0));
	EigenSystem eigensystem = Hamiltonian(0, num_qbits);
	VectorXd initial_phi(dim);
	initial_phi = eigensystem.eigenstates->row(0);

	//MCWF(initial_phi, num_qbits);
	BoltzmanCheck(initial_phi, num_qbits);


	delete eigensystem.H;
        delete eigensystem.eigenstates;
        delete eigensystem.eigenvalues;
	return 0;
}













