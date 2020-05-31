//
// Created by Shrijit Singh on 2020-05-14.
//

#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include <execution/executor.hpp>
#include <execution/property.hpp>

#include "mmul.hpp"

template <typename Executor> void preform_op(Executor &ex) { return; }

int main() {
  auto exec = sse_executor<oneway_t, blocking_t::always_t, void>{}.decay_t();
  std::cout << exec.name() << std::endl;
//
//  auto do_something = [=](int num) -> int { return num; };

  //    auto val = exec.execute(do_something, 4);

  //    std::cout<<val<<std::endl;

//  exec.require(blocking.always);
//
//  std::cout << query(exec, oneway) << std::endl;
//  std::cout << exec.query(oneway) << std::endl;

  preform_op(exec);

  MatrixXd a(3, 3), b(3, 3), c(3, 3);

  a << 1, 1, 1, 1, 1, 1, 1, 1, 1;

  b << 2, 2, 2, 2, 2, 2, 2, 2, 2;

  mmul(inline_executor<oneway_t, blocking_t::always_t, void>{}, a, b, c);
  std::cout<<"Inline: \n"<<c<<std::endl;

  c.setZero();

  mmul(omp_executor<oneway_t, blocking_t::always_t, void>{}, a, b, c);
  std::cout<<"\nOMP: \n"<<c<<std::endl;

}
