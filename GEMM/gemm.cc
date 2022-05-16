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

using namespace std;

int main() {
  std::cout << "------------ GEMM optimization --------------" << std::endl;  
  
 
  std::vector<int> dims = {40, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900};
 
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
    // Deep copy of c 
    auto rawc = CopyMatrix(m, n, c);
    
    // bc
    auto bc = CopyMatrix(m, n, c);
    auto baseline_timecost = benchmark([&](){
      baseline::GEMM(m, n, k, a, b, bc);
    });

    // Copy Result matrix 
    auto result = CopyMatrix(m, n, bc);

    std::cout.setf(ios::fixed); 
    std::cout << std::setprecision(2) << "(m, n, k) = (" << m << ", " << n << ", "<< k << ")" << std::endl;
    std::cout << "GEMM Baseline | timecost = " <<  baseline_timecost << " ms"<< std::endl; 
    
    // oc1
    auto oc1 = CopyMatrix(m, n, c);
    auto timecost = benchmark([&](){
       optim1::GEMM(m, n, k, a, b, oc1);
    });
    std::cout << std::setprecision(2) << "GEMM OPT1     | timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 
    std::cout << "DIFF = " << CompareMatrix(m, n,result, oc1) << std::endl;    
 
    
    auto oc2 = CopyMatrix(m, n, c);
    timecost = benchmark([&](){
       optim2::GEMM(m, n, k, a, b, oc2);
    });
    std::cout << std::setprecision(2) << "GEMM OPT2     | timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 
    std::cout << "DIFF = " << CompareMatrix(m,n,result, oc2) << std::endl;    


    auto oc3 = CopyMatrix(m, n, c);
    timecost = benchmark([&](){
       optim3::GEMM(m, n, k, a, b, oc3);
    });
    std::cout << std::setprecision(2) << "GEMM OPT3     | timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 
    std::cout << "DIFF = " << CompareMatrix(m,n,result, oc3) << std::endl;    

    auto oc4 = CopyMatrix(m, n, c);
    timecost = benchmark([&](){
       optim4::GEMM(m, n, k, a, b, oc4);
    });
    std::cout << std::setprecision(2) << "GEMM OPT4     | timecost = " <<  timecost << " ms"<< "; speedup = "<< baseline_timecost / timecost << "x" << std::endl; 
    std::cout << "DIFF = " << CompareMatrix(m,n,result, oc4) << std::endl;    



    std::cout << "------------------------------------------------" << std::endl; 
   }
}
