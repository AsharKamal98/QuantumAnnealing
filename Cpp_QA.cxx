#include <iostream>
#include <string>
#include <cstring>
#include <bitset>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "eigen-3.4.0/unsupported/Eigen/CXX11/Tensor"
//#include "eigen-3.4.0/unsupported/Eigen/KroneckerProduct"
using namespace std;
using namespace Eigen;

Matrix<int, Dynamic, Dynamic> Hamiltonian_sz(int n) {
	char bit_val1 = '1';

	int dim = pow(2,n);
	MatrixXi H_sz(dim,dim);
	H_sz.setZero();

	for (int index=0; index<dim; index++) {
		string bin_index = bitset<3>(index).to_string();	// BITSET REQUIRES SYSTEM SIZE DEFINITION AT COMPILE TIME
		
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
                string bin_index = bitset<3>(index).to_string();	// BITSET REQUIRES SYSTEM SIZE DEFINITION AT COMPILE TIME
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


//Matrix<double, Dynamic, Dynamic> Hamiltoneon(double s, int n) {
MatrixXd Hamiltonian(int s, int n) {

	cout << "Hello World\n";
	struct {
		Matrix<double, Dynamic, Dynamic> H;
		Matrix<double, Dynamic, Dynamic> eigenvectors;
		Matrix<double, Dynamic, 1> eigenvalues;
	} eigensystem;

	//Tensor<int, 3> I(1,2,2), sx(1,2,2), sz(1,2,2);
	//I.setValues({{{1,0}, {0,1}}});
	//sz.setValues({{{1,0}, {0,-1}}});
        //sx.setValues({{{0,1}, {1,0}}});
	
	int dim = pow(2,n);
	//Tensor<float, 3> H_sub(2, dim, dim);
	//Tensor<int, 3> pauli(2, 2, 2);
	//pauli.slice(Eigen::array<Index, 3>{{0, 0, 0}}, Eigen::array<Index, 3>{{1, 2, 2}}) = sx;
	//pauli.slice(Eigen::array<Index, 3>{{1, 0, 0}}, Eigen::array<Index, 3>{{1, 2, 2}}) = sz;	

	//Tensor<int,3> op_list(n,2,2);
	//for (int i=0; i<2; i++) {
	//	Tensor<int,3> s = pauli.slice(Eigen::array<Index, 3>{{i, 0, 0}}, Eigen::array<Index, 3>{{1, 2, 2}});
	//	for (int j=0; j<n; j++) {
	//		op_list = I.broadcast(Eigen::array<Index,3>{{n,1,1}});
	//		op_list.slice(Eigen::array<Index,3>{{j,0,0}}, Eigen::array<Index,3>{{1,2,2}}) = s;

	//		Tensor<int,2> mat1 = op_list.slice(Eigen::array<Index,3>{{0,0,0}}, Eigen::array<Index,3>{{1,2,2}}).reshape(Tensor<int,2>::Dimensions(2,2));
	//		Tensor<int,2> mat2 = op_list.slice(Eigen::array<Index,3>{{1,0,0}}, Eigen::array<Index,3>{{1,2,2}}).reshape(Tensor<int,2>::Dimensions(2,2));
			
			//Tensor<int,2> kron_product = KroneckerProduct(mat1,mat2);
			//int val = kron_product(3,3);
			//cout << kron_product << "\n\n";
	//	}	
	//}

	//for (int i=0; i<n; i++) {
	//	Tensor<int,2> slice = op_list.slice(Eigen::array<Index,3>{{i,0,0}}, Eigen::array<Index,3>{{1,2,2}}).reshape(Tensor<int,2>::Dimensions(2,2)); 
	//	cout << slice << "\n";
	//}

	//MatrixXd H(dim,dim);
	eigensystem.H.setZero();

	MatrixXi H_sz = Hamiltonian_sz(n);
	MatrixXi H_sx = Hamiltonian_sx(n);
    	//s1.diagonal() = VectorXd::Constant(dim, s);

	for (int i=0; i<dim; i++) {
		for (int j=0; j<dim; j++) {
	//		eigensystem.H(i,j) = H_sx(i,j) * (1-s) + H_sz(i,j) * s;
	//	}
	//}

	//SelfAdjointEigenSolver<MatrixXd> eigensolver;
	//eigensolver.compute(eigensystem.H);
	//eigensystem.eigenvectors = eigensolver.eigenvectors();
	//eigensystem.eigenvalues << eigensolver.eigenvalues();


	//cout << eigensystem.eigenvectors << "\n";

	//cout << eigenvectors << "\n";
	//cout << eigenvalues << "\n";
	//cout << H << "\n";

	return eigensystem.H;
}





int main() {
	MatrixXd H, J;
        H = Hamiltonian(1, 3);
	//cout << H << "\n";
	return 0;
}














