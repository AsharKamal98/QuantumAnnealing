#include <iostream>
#include <string>
#include <cstring>
#include <bitset>
#include <cstdlib>
#include <time.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "eigen-3.4.0/unsupported/Eigen/CXX11/Tensor"
//#include "eigen-3.4.0/unsupported/Eigen/KroneckerProduct"
using namespace std;
using namespace Eigen;

const int num_qbits = 2;
const int dim = pow(2,num_qbits);
const double T = 0.1;
const double lam_sq = 1;

double PickRandomWeightedElement(const std::vector<double>& weights) {
	int total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
	double random_num = (static_cast<double>(std::rand()) / RAND_MAX)  * total_weight;	

	double cumulative_sum = 0;
	for (std::size_t i=0; i<weights.size(); i++) {
		cumulative_sum += weights[i];
		if (random_num < cumulative_sum) {
			return i;
		}
	}

	return -1;
}


double Round(const double value, const double precision) {
	return round(value/precision) * precision;
}

Matrix<int, Dynamic, Dynamic> Hamiltonian_sz(const int n) {
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


Matrix<int, Dynamic, Dynamic> Hamiltonian_sx(const int n) {
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


EigenSystem Hamiltonian(const double s, const int n) {
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


void MCWF(VectorXcd phi, const int n) {
	cout << "\nInitial phi \n" << phi.transpose() << "\n\n";

	const double dt_ds_ratio = 2;
	const double ds = 0.000025;
	const double dt = dt_ds_ratio * ds;
	const int iterations_s = static_cast<int>(1/ds);
	const int iterations_t = 10;

	VectorXcd phi_history_z(iterations_s);
	VectorXcd phi_history_x(iterations_s);
	VectorXcd phi_decomp(dim);
	VectorXd eigenval_temp(dim);

	for (int i=0; i<=iterations_s; i++) {
		//cout << "----ITERATION-----: " << i << "\n";
		double s = i * ds;
		//double s = 1;
		EigenSystem eigensystem = Hamiltonian(s, n);
		phi_decomp = *eigensystem.eigenstates * phi.conjugate();

		// REMOVE	
		//cout << "Hamiltonian\n" << *eigensystem.H << "\n";
		//cout << "eigenstates\n" << *eigensystem.eigenstates << "\n";
		//cout << "eigenvalues: " << (*eigensystem.eigenvalues).transpose() << "\n\n";
		for (int k=0; k<4; k++) {
			double eigen_norm = (*eigensystem.eigenstates).row(k).norm();
			if (eigen_norm < 0.98 or eigen_norm > 1.02) {
				cout << "############ EIGENSTATES NOT NORMALIZED ##############\n";
				cout << "eigenstate norm: " << eigen_norm << "\n";
			}
		}
		if (phi_decomp.norm() - phi.norm() > 0.01) {
			cout << "############### PHI_DECOMP AND PHI DIFFERENT NORMS ##############\n";
			cout << "phi_decomp norm: " << phi_decomp.norm() << "\n";
			cout << "phi norm: " << phi.norm() << "\n";
		}
		//cout << "phi norm: " << phi.norm() << "\n";
		//cout << "phi_decomp norm: " << phi_decomp.norm() << "\n";


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
					if (Round(energy_a-energy_b, 0.0001) > 0) {
						pre_factors.push_back((N(energy_a, energy_b)+1)*lam_sq);
						photon_type = -1;
					// Absorption (energy_a < energy_b)
					} else if (Round(energy_b-energy_a, 0.0001) > 0) {
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
					complex<double> temp1 = phi_decomp.conjugate().transpose() * C(b,a, C(a,b,phi_decomp));
					double temp2 = temp1.real();
					delta_p_list.push_back(pre_factors.back() * temp2 * dt);

					//cout << "Transition: " << a << b << "\n";
					//cout << "photon type: " << photon_type << "\n";
					//cout << "pre_factor: " << pre_factors.back() << "\n\n";


					if (delta_p_list.back() > 1) {
						cout << "delta_p_list: " << delta_p_list.back() << "\n";
						cout << "pre_factors: " << pre_factors.back() << "\n";
						cout << "temp: " << temp2 << "\n";
						cout << "energy_a: " << energy_a << "\n";
						cout << "energy_b: " << energy_b << "\n\n";	
					}
					//cout << "PRE_FACTOR1: " << pre_factors.back() << "\n";
				}
			}
			double delta_p = std::accumulate(delta_p_list.begin(), delta_p_list.end(), 0.0);
			double epsilon = (static_cast<double>(std::rand()) / RAND_MAX);
			
			if (delta_p > 0.1) {
				cout << "Warning! delta_p  is getting large, must be much smaller than 1. Current value:" << delta_p << "\n";
			}

			
			VectorXcd phi_new(dim);
			// No emission/absorption
			//cout << "phi_decomp before " << phi_decomp.array().abs2().transpose() << "\n";
			if (true) { //(epsilon > delta_p) {
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
						//cout << "a, b: " << a << b << "\n";
						//cout << "phi_decomp " << phi_decomp.transpose() << "\n";
						//cout << "phi_1 " << pre_factors[counter] * C(b,a, C(a,b,phi_decomp)).transpose() << "\n";
						//cout << "pre_factor " << pre_factors[counter] << "\n\n";
						//cout << "PRE_FACTOR2: " << pre_factors[counter]	<< "\n";
						counter++;

					}
				}
				//cout << "phi_1 " << phi_1.array().abs2().transpose() << "\n";
				phi_new = phi_1/pow(1-delta_p,0.5);
				//cout << "phi_new " << phi_new.array().abs2().transpose() << "\n";
			// Emission/absorption
			} else {
				size_t index =  PickRandomWeightedElement(delta_p_list);	
				double delta_p_m = delta_p_list[index];
				int a = counter_list1[index]; int b = counter_list1[index+1];
				int photon_type = counter_list2[index];
				if (photon_type==-1) {
					cout << "Spontaneous emission\n";
				} else if (photon_type==1) {
					cout << "Absorption!\n";
				}

				phi_new = pow(pre_factors[index], 0.5) * (C(a,b,phi_decomp)/pow(delta_p_m/dt,0.5)); 		

			}
			phi_decomp = phi_new;
			//cout << "phi_decomp after " << phi_decomp.array().abs2().transpose() << "\n";
			//cout << "eigenvalues " << (*eigensystem.eigenvalues).transpose() << "\n";

			//cout << "phi_new norm: " << phi_new.norm() << "\n";

			//cout << "iteration: " << i << "\n";
			//cout << "delta_p: " << delta_p << "\n";
			//cout << "delta_p_list:  ";
			//for (int k=0; k<delta_p_list.size(); k++) {
			//	cout << delta_p_list[k] << "  ";
			//}
			//cout << "\nphi norm: " << Round(phi_decomp.norm(), 0.01) << "\n\n";

		}
		
		// Back to old basis
		phi = (*eigensystem.eigenstates).transpose() * phi_decomp;
		//cout << "phi after " << phi.array().abs2().transpose() << "\n\n";	
		eigenval_temp = *eigensystem.eigenvalues;

		delete eigensystem.H;
        	delete eigensystem.eigenstates;
        	delete eigensystem.eigenvalues;

	}


	double precision = 0.01;
	cout << "\n\n------------- SUMMARY ------------";
	cout << "\nphi\n" << phi.transpose().array() << "\n";
	cout << "phi_decomp\n" << phi_decomp.transpose() << "\n";
	cout << "real phi_decomp\n" << round((phi_decomp.transpose().array()).abs2()/precision) * precision << "\n";
	cout << "phi norm: " << Round(phi_decomp.norm(), precision) << "\n\n";
	cout << "eigenvalues: " << eigenval_temp.transpose() << "\n";

}


int main() {
	srand(time(0));
	EigenSystem eigensystem = Hamiltonian(0, num_qbits);
	VectorXd initial_phi(dim);
	initial_phi = eigensystem.eigenstates->row(0);

	MCWF(initial_phi, num_qbits);


	delete eigensystem.H;
        delete eigensystem.eigenstates;
        delete eigensystem.eigenvalues;
	return 0;
}













