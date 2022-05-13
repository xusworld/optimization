#include "common.h"


namespace optim2 {

// Create macro to let X(i) equal the ith element of x 
#define X(i) x[(i) * incx]

void AddDot1x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes four elements of C: 

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i, j ), C( i, j+1 ), C( i, j+2 ), C( i, j+3 ) 
          
     in the original matrix C.

     In this version, we merge the four loops, computing four inner
     products simultaneously. */

  int p;

  //  AddDot( k, &A( 0, 0 ), lda, &B( 0, 0 ), &C( 0, 0 ) );
  //  AddDot( k, &A( 0, 0 ), lda, &B( 0, 1 ), &C( 0, 1 ) );
  //  AddDot( k, &A( 0, 0 ), lda, &B( 0, 2 ), &C( 0, 2 ) );
  //  AddDot( k, &A( 0, 0 ), lda, &B( 0, 3 ), &C( 0, 3 ) );
  for (p = 0; p < k; ++p){
    C( 0, 0 ) += A( 0, p ) * B( p, 0 );     
    C( 0, 1 ) += A( 0, p ) * B( p, 1 );     
    C( 0, 2 ) += A( 0, p ) * B( p, 2 );     
    C( 0, 3 ) += A( 0, p ) * B( p, 3 );     
  }
}

void GEMM(int m, int n, int k, double *a, double* b, double* c) { 
  int i, j;

  // Loop over the columns of C
  for (j = 0; j < n; j += 4) {    
    // Loop over the rows of C   
    for (i = 0; i < m; ++i) {      
      //  Update the C( i,j ) with the inner product of the ith row of A and the jth column of B  
      AddDot1x4(k, &A(i, 0), sa, &B(0, j), sb, &C(i, j), sc);
    }
  }
}

} // namespace optim2
