#include "common.h"


namespace baseline {

void GEMM(int m, int n, int k, double* a, double* b, double* c) { 
  int i, j, p;

  for (i=0; i < m; ++i) {        
    for (j=0; j < n; ++j) {
      for (p=0; p < k; ++p) {  
        c[i*n + j] += a[i*k + p] * b[p*n + j]; 
      }
    }
  }
}

} // namespace baseline
