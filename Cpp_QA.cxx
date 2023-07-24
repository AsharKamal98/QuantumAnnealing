#include <iostream>
#include <string>
#include <cstring>
#include <bitset>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "eigen-3.4.0/unsupported/Eigen/CXX11/Tensor"
//#include "eigen-3.4.0/unsupported/Eigen/KroneckerProduct"
using namespace std;
using namespace Eigen;

int num_qbits = 2;
int dim = pow(2,num_qbits);


int PickRandomWeightedElement(const std::vector<double>& weights) {
	int total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);



Matrix<int, Dynamic, Dynamic> Hamiltonian_sz(int n) {
	char bit_val1 = '1';

	int dim = pow(2,n);
	MatrixXi H_sz(dim,dim);
	H_sz.setZero();

	for (int index=0; index<dim; index++) {
		string bin_index = bitset<2>(index).to_string();	// BITSET REQUIRES SYSTEM SIZE DEFINITION AT COMPILE TIME
		
		int diag_val = 0;
		for (int bit=0; bit<n; bit++) {
			char bit_val = bin_index[bit];
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


Matrix<int, Dynamic, Dynamic> Hamiltonian_sx(int n) {
        char bit_val1 = '1';

        int dim = pow(2,n);
        MatrixXi H_sx(dim,dim);
        H_sx.setZero();

        for (int index=0; index<dim; index++) {
                string bin_index = bitset<2>(index).to_string();	// BITSET REQUIRES SYSTEM SIZE DEFINITION AT COMPILE TIME
                for (int bit=0; bit<n; bit++) {
			string bin_index_temp = bin_index;
                        char bit_val = bin_index[bit];
                        if (bit_val == bit_val1) {
                        	bin_index_temp[bit] = '0';
			} else {
                        	bin_index_temp[bit] = '1';
			}

			bitset<10> b(bin_index_temp);
			int flipped_index = b.to_ulong();
			H_sx(index,flipped_index) = 1;
                }
        }
        return H_sx;
}

struct EigenSystem {
		MatrixXd* H;
		MatrixXd* eigenstates;
		VectorXd* eigenvalues;
	};


EigenSystem Hamiltonian(double s, int n) {
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
	//cout << *eigensystem.eigenvalues << "\n";
	//cout << *eigensystem.H << "\n";

	return eigensystem;
}

VectorXcd C(int n, int a, int b, VectorXcd phi) {
	VectorXcd phi_prime(dim);
	phi_prime.setZero();
	phi_prime(b) = phi(a);
	return phi_prime;
}


void MCWF(VectorXcd phi, int n) {
	cout << "Initial phi \n" << phi.transpose() << "\n\n";

	int dim = static_cast<int>(pow(2,n));	
	double dt_ds_ratio = 2;
	double ds = 0.5;
	double dt = dt_ds_ratio * ds;
	int iterations_s = static_cast<int>(1/ds);
	int iterations_t = 1;

	VectorXcd phi_history_z(iterations_s);
	VectorXcd phi_history_x(iterations_s);


	for (int i=0; i<=iterations_s; i++) {
		double s = i * ds;
		EigenSystem eigensystem = Hamiltonian(s, n);
		VectorXcd phi_decomp = (*eigensystem.eigenstates) * phi.conjugate();
		double phi_decomp_norm = phi_decomp.norm();
		
		for (int j=0; j<iterations_t; j++) {
			std::vector<double> pre_factors;
			std::vector<double> counter_list1;
			std::vector <char> counter_list2;
			std::vector<double> energy_list;
			std::vector<double> delta_p_list;
			int photon_type; //1 for abs, -1 for emiss and 0 for none
					 
			for (int a=0; a<dim; a++) {
				double energy_a = (*eigensystem.eigenvalues)(a);
				for (int b=0; b<dim; b++) {
					double energy_b = (*eigensystem.eigenvalues)(b);
					if (a==b) { 
						continue;
					}
					// Spontaneous emission
					if (energy_a > energy_b) {
						pre_factors.push_back(1);
						photon_type = -1;
					// Absorption
					} else if (energy_a < energy_b) {
						pre_factors.push_back(-1);
						photon_type = 1;
					// Degenerate eigenvalues (energy_a = energy_b)
					} else {
						pre_factors.push_back(0);
						photon_type = 0;	
					}
					counter_list1.push_back(a);
					counter_list1.push_back(b);
					counter_list2.push_back(photon_type);
					energy_list.push_back(energy_a);
					energy_list.push_back(energy_b);
					complex<double> temp1 = phi_decomp.conjugate().transpose() * C(n,b,a, C(n,a,b,phi_decomp));
					double temp2 = temp1.real();
					delta_p_list.push_back(pre_factors.back() * temp2 * dt); 
				}
			}
			double delta_p = std::accumulate(delta_p_list.begin(), delta_p_list.end(), 0.0);
			double epsilon = (double) rand()/RAND_MAX;
		
		if (delta_p > 0.1) {
			cout << "Warning! delta_p  is getting large, must be much smaller than 1. Current value:" << delta_p << "\n";
		}

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
				phi_1 -= (1/2) * pre_factors[counter] * C(n,b,a, C(n,a,b,phi_decomp)) * dt;
				counter++;
				}
			}
			VectorXcd phi_new = phi_1/pow(1-delta_p,0.5);
		} else {
			//here!	
		
		}	
		



		delete eigensystem.H;
        	delete eigensystem.eigenstates;
        	delete eigensystem.eigenvalues;
	}



}


int main() {
	//int num_qbits = 2;
	//int dim = pow(2,num_qbits);
	double T = 1;
	double lam_sq = 1;

	EigenSystem eigensystem = Hamiltonian(0, num_qbits);
	VectorXd initial_phi(dim);
	initial_phi = eigensystem.eigenstates->col(0);

	MCWF(initial_phi, num_qbits);



	delete eigensystem.H;
        delete eigensystem.eigenstates;
        delete eigensystem.eigenvalues;
	return 0;
}













