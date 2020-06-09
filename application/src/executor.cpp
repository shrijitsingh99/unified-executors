//
// Created by Shrijit Singh on 2020-05-14.
//

#include <iostream>

#include <Eigen/Dense>

#include <mmul.hpp>

using namespace Eigen;

int main() {
  double dataA[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9},
         dataB[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9}, dataC[9] = {0};
  MatrixXd a = Map<Matrix<double, 3, 3, RowMajor>>(dataA);
  MatrixXd b = Map<Matrix<double, 3, 3, RowMajor>>(dataB);
  MatrixXd c = Map<Matrix<double, 3, 3, RowMajor>>(dataC);

  mmul(inline_executor<oneway_t, single_t, blocking_t::always_t, void>{}.decay_t(), a, b, c);
  std::cout<<"Inline: \n"<<c<<std::endl;

  c.setZero();
  mmul(omp_executor<oneway_t, bulk_t, blocking_t::always_t, void>{}, a, b, c);
  std::cout<<"\nOMP: \n"<<c<<std::endl;

  c.setZero();
  mmul(cuda_executor<oneway_t, bulk_t, blocking_t::always_t, void>{}.decay_t(), a, b, c);
  std::cout<<"\nCUDA: \n"<<c<<std::endl;

}
