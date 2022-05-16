#include "common.h"


namespace optim1 {

/*  
v1 no speedup
void InnerKernel4x4(int k, double* a, int sa, double* b, int sb, double* c, int sc) {
  int i, j, p;
  for (i = 0; i < 4; i++) {
    for(j = 0; j < 4; j++) {
      for(p = 0; p < k; p++) {
        C(i, j) += A(i, p) * B(p, j);
      }
    }
  }
}

*/

/* 
v2 max 1.4x speedup

void InnerKernel4x4(int k, double* a, int sa, double* b, int sb, double* c, int sc) {
  int i, j, p;
  
  for (i = 0; i < 4; i++) {
    for(p = 0; p < k; p++) {
      C(i, 0) += A(i, p) * B(p, 0);
      C(i, 1) += A(i, p) * B(p, 1);
      C(i, 2) += A(i, p) * B(p, 2);
      C(i, 3) += A(i, p) * B(p, 3);
    }
  }
}

*/

void InnerKernel4x4(int k, double* a, int sa, double* b, int sb, double* c, int sc) {
  int p;
  
  for(p = 0; p < k; p++) {
    C(0, 0) += A(0, p) * B(p, 0);
    C(0, 1) += A(0, p) * B(p, 1);
    C(0, 2) += A(0, p) * B(p, 2);
    C(0, 3) += A(0, p) * B(p, 3);

    C(1, 0) += A(1, p) * B(p, 0);
    C(1, 1) += A(1, p) * B(p, 1);
    C(1, 2) += A(1, p) * B(p, 2);
    C(1, 3) += A(1, p) * B(p, 3);

    C(2, 0) += A(2, p) * B(p, 0);
    C(2, 1) += A(2, p) * B(p, 1);
    C(2, 2) += A(2, p) * B(p, 2);
    C(2, 3) += A(2, p) * B(p, 3);

    C(3, 0) += A(3, p) * B(p, 0);
    C(3, 1) += A(3, p) * B(p, 1);
    C(3, 2) += A(3, p) * B(p, 2);
    C(3, 3) += A(3, p) * B(p, 3);
  }
}

void GEMM(int m, int n, int k, double* a, double* b, double* c) { 
  int i, j, p;

  for (i=0; i < m; i+=4) {        
    for (j=0; j < n; j+=4) {
      InnerKernel4x4(k, &A(i, 0), sa, &B(0, j), sb, &C(i, j), sc);
    }
  }
}

} // namespace optim1 
