//
// Created by Shrijit Singh on 2020-05-14.
//

#include <iostream>

#include <execution/executor.hpp>
#include <execution/property.hpp>

int main() {
    auto exec = sse_executor<blocking_t::always_t>{}.decay_t();
    std::cout<<exec.name()<<std::endl;

    auto do_something = [=](int num)-> int {
      return num;
    };

    auto val = exec.execute(do_something, 4);

    std::cout<<val<<std::endl;

    exec.require(blocking.always);

}
