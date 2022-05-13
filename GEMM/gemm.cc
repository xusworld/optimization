#include <iostream>
#include <vector>
#include <random>
#include <functional>

#include "util.h"
#include "baseline.hpp"
#include "optim1.hpp"
#include "optim2.hpp"
#include "optim3.hpp"

using namespace std;

int main() {
  std::cout << "------------ GEMM optimization --------------" << std::endl;  
  
 
  std::vector<int> dims = {40, 80, 100, 120, 160, 200, 240, 300, 320,  400, 440, 500};
 
  for (auto dim : dims) {
    int m = dim;
    int n = dim;
    int k = dim;

    sa = m;
    sb = k;
    sc = n;

    auto a = RandomMatrix(m, k);
    auto b = RandomMatrix(k, n);
    auto c = RandomMatrix(m, n);

    auto baseline_timecost = benchmark([&](){
      baseline::GEMM(m, n, k, a, b, c);
    });

    std::cout << "(m, n, k) = (" << m << " ," << n << " ,"<< k << ")" << std::endl;

    std::cout << "GEMM Baseline ,  timecost = " <<  baseline_timecost << " ms"<< std::endl; 


    auto timecost = benchmark([&](){
      optim1::GEMM(m, n, k, a, b, c);
    });
    std::cout << "GEMM OPT1     , timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 

    timecost = benchmark([&](){
      optim2::GEMM(m, n, k, a, b, c);
    });
    std::cout << "GEMM OPT2     , timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 

    timecost = benchmark([&](){
      optim3::GEMM(m, n, k, a, b, c);
    });
    std::cout << "GEMM OPT3     , timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 


    std::cout << "------------------------------------------------" << std::endl; 
   }
}
