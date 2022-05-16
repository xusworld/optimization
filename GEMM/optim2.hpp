#include "common.h"


namespace optim2 {


#define BLOCK 128


#define min( i, j ) ( (i)<(j) ? (i): (j) )

void AddDot4x4(int k, double* a, int sa, double* b, int sb, double* c, int sc) {
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

void InnerKernel(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc) {
  int i, j;

  for (i=0; i < m; i+=4) {        
    for (j=0; j < n; j+=4) {
      AddDot4x4(k, &A(i, 0), sa, &B(0, j), sb, &C(i, j), sc);
    }
  }

}

void GEMM(int m, int n, int k, double* a, double* b, double* c) { 
  int i, j, p, pb, ib;
  for (i = 0; i < m; i+=BLOCK) {
    ib = min(m-i, BLOCK); 

    for (p = 0; p < k; p+=BLOCK) {
      pb = min(k-p, BLOCK);
      InnerKernel(ib, n, pb, &A(i, p), sa, &B(p, 0), sb, &C(i, 0), sc);
    }
  }
}

} // namespace optim2 
