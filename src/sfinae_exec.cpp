//
// Created by Shrijit Singh on 2020-05-14.
//

#include <iostream>

#include <sfinae_prop.hpp>
#include <sfinae_exec.hpp>

int main() {
  auto exec = sse_executor{}.fallback_t();
  std::cout<<exec.name()<<std::endl;

  auto do_something = [=](int num)-> int {
    return num;
  };

  auto val = exec.execute(do_something, 4);

  std::cout<<val;

  exec.require(blocking_t::always_t{});
//  exec.require(blocking_t::never_t{}); Won't Compile
}
