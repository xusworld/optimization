#include "common.h"


namespace optim3 {


#define BLOCK 128


#define min( i, j ) ( (i)<(j) ? (i): (j) )

//#include <immintrin.h>  // AVX AVX2 FMA
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

typedef union
{
  __m128d v;
  double d[2];
} Vec2;

void AddDot4x4(int k, double* a, int sa, double* b, int sb, double* c, int sc) {
  int p;
 

  Vec2 c_00_c_01_vreg, c_02_c_03_vreg,
       c_10_c_11_vreg, c_12_c_13_vreg,
       c_20_c_21_vreg, c_22_c_23_vreg,
       c_30_c_31_vreg, c_32_c_33_vreg;
  
  double *a0p_pntr, *a1p_pntr, *a2p_pntr, *a3p_pntr; 
  Vec2 a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg;

  Vec2 b_p0_b_p1_vreg, b_p2_b_p3_vreg;


  a0p_pntr = &A(0, 0);
  a1p_pntr = &A(1, 0);
  a2p_pntr = &A(2, 0);
  a3p_pntr = &A(3, 0);

  c_00_c_01_vreg.v = _mm_setzero_pd();
  c_02_c_03_vreg.v = _mm_setzero_pd();
  c_10_c_11_vreg.v = _mm_setzero_pd();
  c_12_c_13_vreg.v = _mm_setzero_pd();
  c_20_c_21_vreg.v = _mm_setzero_pd();
  c_22_c_23_vreg.v = _mm_setzero_pd();
  c_30_c_31_vreg.v = _mm_setzero_pd();
  c_32_c_33_vreg.v = _mm_setzero_pd();

  for(p = 0; p < k; ++p) {
    // Load B(p, 0), B(p, 1), B(p, 2) and B(p, 3)
    b_p0_b_p1_vreg.v = _mm_load_pd((double *)&B(p, 0)); 
    b_p2_b_p3_vreg.v = _mm_load_pd((double *)&B(p, 2));

    double tmp[2];
    _mm_storeu_pd(tmp, b_p0_b_p1_vreg.v); 
    double tmp1[2];
    _mm_storeu_pd(tmp1, b_p2_b_p3_vreg.v); 

    // Load A(0, p), A(1, p), A(2, p) and A(3, p)
    a_0p_vreg.v = _mm_loaddup_pd((double *)a0p_pntr++);
    a_1p_vreg.v = _mm_loaddup_pd((double *)a1p_pntr++);
    a_2p_vreg.v = _mm_loaddup_pd((double *)a2p_pntr++);
    a_3p_vreg.v = _mm_loaddup_pd((double *)a3p_pntr++);

    // First row
    c_00_c_01_vreg.v += a_0p_vreg.v * b_p0_b_p1_vreg.v;
    c_02_c_03_vreg.v += a_0p_vreg.v * b_p2_b_p3_vreg.v;
    // Second row
    c_10_c_11_vreg.v += a_1p_vreg.v * b_p0_b_p1_vreg.v;
    c_12_c_13_vreg.v += a_1p_vreg.v * b_p2_b_p3_vreg.v;
    // Third row 
    c_20_c_21_vreg.v += a_2p_vreg.v * b_p0_b_p1_vreg.v;
    c_22_c_23_vreg.v += a_2p_vreg.v * b_p2_b_p3_vreg.v;
    // Fourth row
    c_30_c_31_vreg.v += a_3p_vreg.v * b_p0_b_p1_vreg.v;
    c_32_c_33_vreg.v += a_3p_vreg.v * b_p2_b_p3_vreg.v;
  }
 
  C(0, 0) += c_00_c_01_vreg.d[0];
  C(0, 1) += c_00_c_01_vreg.d[1];
  C(0, 2) += c_02_c_03_vreg.d[0];
  C(0, 3) += c_02_c_03_vreg.d[1];
 
  C(1, 0) += c_10_c_11_vreg.d[0];
  C(1, 1) += c_10_c_11_vreg.d[1];
  C(1, 2) += c_12_c_13_vreg.d[0];
  C(1, 3) += c_12_c_13_vreg.d[1];
 
  C(2, 0) += c_20_c_21_vreg.d[0];
  C(2, 1) += c_20_c_21_vreg.d[1];
  C(2, 2) += c_22_c_23_vreg.d[0];
  C(2, 3) += c_22_c_23_vreg.d[1];
 
  C(3, 0) += c_30_c_31_vreg.d[0];
  C(3, 1) += c_30_c_31_vreg.d[1];
  C(3, 2) += c_32_c_33_vreg.d[0];
  C(3, 3) += c_32_c_33_vreg.d[1];

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

} // namespace optim3 
