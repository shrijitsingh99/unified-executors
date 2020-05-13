//
// Created by Shrijit Singh on 2020-05-14.
//

#include <iostream>

#include <sfinae_exec.h>


int main() {
  sse_executor exec;
  std::cout<<exec().name()<<std::endl;
}
