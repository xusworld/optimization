#include "common.h"


namespace optim1 {

// Create macro to let X(i) equal the ith element of x 
#define X(i) x[(i) * incx]

void AddDot( int k, double *x, int incx,  double *y, double *gamma ) {
  /* 
  compute gamma := x' * y + gamma with vectors x and y of length n.
  Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
  */
  for (int p=0; p < k; ++p){
    *gamma += X(p) * y[p];     
  }
}

void GEMM(int m, int n, int k, double *a, double* b, double* c) { 
  int i, j;

  // Loop over the columns of C
  for (j = 0; j < n; ++j) {    
    // Loop over the rows of C   
    for (i = 0; i < m; ++i) {      
      //  pdate the C( i,j ) with the inner product of the ith row of A and the jth column of B  
      AddDot(k, &A(i, 0), sa, &B(0, j), &C(i, j));
    }
  }
}

} // namespace optim1
