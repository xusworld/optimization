#include <iostream>
#include <vector>
#include <functional>
#include "util.h"


using namespace std;


#define A(i,j) a[ (j)*sa + (i) ]
#define B(i,j) b[ (j)*sb + (i) ]
#define C(i,j) c[ (j)*sc + (i) ]

int sa;
int sb;
int sc;

void GEMM(int m, int n, int k, double *a, double* b, double* c) { 
  int i, j, p;

  for ( i=0; i<m; i++ ) {        
    for ( j=0; j<n; j++ ) {     
      for ( p=0; p<k; p++ ) {  
	C(i,j) = C(i,j) +  A(i, p) * B(p, j);
      }
    }
  }
}


int main() {
  std::cout << "------------ GEMM optimization --------------" << std::endl;  
  
 
  std::vector<int> dims = {16, 32, 64, 128, 256};
 
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

    auto timecost = benchmark([&](){
      GEMM(m, n, k, a, b, c);
    });

    std::cout << "GEMM Baseline , (m, n, k) = (" << m << " ," << n << " ,"<< k << "), timecost = " <<  timecost << " ms"<< std::endl; 
    std::cout << "------------------------------------------------" << std::endl; 
   }
}
