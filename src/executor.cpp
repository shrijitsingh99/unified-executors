//
// Created by Shrijit Singh on 2020-05-14.
//

#include <iostream>
#include <vector>

#include <execution/executor.hpp>
#include <execution/property.hpp>

int main() {
    auto exec = sse_executor<oneway_t, blocking_t::always_t, void>{}.decay_t();
    std::cout<<exec.name()<<std::endl;

    auto do_something = [=](int num)-> int {
      return num;
    };

    auto val = exec.execute(do_something, 4);

    std::cout<<val<<std::endl;

    exec.require(blocking.always);

    std::cout<<query(exec, oneway)<<std::endl;
    std::cout<<exec.query(oneway)<<std::endl;
}
