#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <iomanip>

#include "util.h"
#include "baseline.hpp"
#include "optim1.hpp"
#include "optim2.hpp"
#include "optim3.hpp"
#include "optim4.hpp"
#include "optim5.hpp"

using namespace std;

int main() {
  std::cout << "------------ GEMM optimization --------------" << std::endl;  
  
 
  std::vector<int> dims = {40, 80, 100, 200, 300, 400, 500};
 
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

    std::cout.setf(ios::fixed); 
    std::cout << std::setprecision(2) << "(m, n, k) = (" << m << ", " << n << ", "<< k << ")" << std::endl;

    std::cout << "GEMM Baseline | timecost = " <<  baseline_timecost << " ms"<< std::endl; 


    auto timecost = benchmark([&](){
      optim1::GEMM(m, n, k, a, b, c);
    });
    std::cout << std::setprecision(2) << "GEMM OPT1     | timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 

    timecost = benchmark([&](){
      optim2::GEMM(m, n, k, a, b, c);
    });
    std::cout << std::setprecision(2) <<  "GEMM OPT2     | timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 

    timecost = benchmark([&](){
      optim3::GEMM(m, n, k, a, b, c);
    });
    std::cout << std::setprecision(2) <<  "GEMM OPT3     | timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 
    
    /*
    timecost = benchmark([&](){
      optim4::GEMM_UNLOOP(m, n, k, a, b, c);
    });
    std::cout << std::setprecision(2) <<  "GEMM OPT4(UNLOOP)     | timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 
    */

    timecost = benchmark([&](){
      optim4::GEMM(m, n, k, a, b, c);
    });
    std::cout << std::setprecision(2) <<  "GEMM OPT4     | timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 

    timecost = benchmark([&](){
      optim5::GEMM(m, n, k, a, b, c);
    });
    std::cout << std::setprecision(2) <<  "GEMM OPT5     | timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 


    std::cout << "------------------------------------------------" << std::endl; 
   }
}
