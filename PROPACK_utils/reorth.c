/* 
 * This is a .c implementation of Larsen's reorth.f  
 * by Stephen Becker, 3/7/09
*/

#include <stdio.h>
#include <math.h>
#include "mex.h"
#ifndef NO_BLAS
    #include "blas.h"
#endif


#ifndef blas_h
    #define blas_h

    #if !defined(_WIN32) 
    #define dgemm dgemm_ 
    #define dnrm2 dnrm2_ 
    #endif 

    #ifdef WINDOWS 
      extern void dgemv(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA, 
              double *X, int *INCX, double *BETA, double *Y, int *INCY); 
      extern double dnrm2(int *N, double *X, int *INCX); 
    #else 
      void dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA, 
              double *X, int *INCX, double *BETA, double *Y, int *INCY); 
      extern double dnrm2_(int *N, double *X, int *INCX);  
      double dnrm2_(int *N, double *X, int *INCX); 
    #endif 
#endif 


/* Modified Gram Schmidt re-orthogonalization */
void MGS( int *n, int *k, double *V, int *ldv, double *vnew, double *index ) {
    int i, j, idx;
    int LDV = *ldv;
    double s;
    for (i=0;i<*k;i++){
        idx = (int) index[i]-1;  /* -1 because MATLAB uses 1-based indices */
        s = 0.0;
        for (j=0;j<*n;j++)
            /*s += V[j,idx]*vnew[j]; */ 
            s += V[ idx*LDV + j ] * vnew[j];
        for (j=0;j<*n;j++)
            /* vnew[j] -= s*V[j,idx]; */ /* Fortran is row-indexed */
            vnew[j] -= s*V[ idx*LDV + j ];
    }
}

void reorth( int *n, int *k, double *V, int *ldv, double *vnew, double *normv, double *index,
        double *alpha, double *work, int *iflag, int *nre ) {

    int i;
    ptrdiff_t one = 1, N = (ptrdiff_t) *n, K = (ptrdiff_t) *k;
    ptrdiff_t LDV = (ptrdiff_t) *ldv;
    char Transpose = 'T', Normal = 'N';
    double normv_old;
    const int MAXTRY = 4;

    double oneD = 1.0, nOneD = -1.0, zero = 0.0;

#ifdef WINDOWS
    void (*dgemvPtr)(char *, ptrdiff_t *, ptrdiff_t *, double *, double *, ptrdiff_t *, 
          double *, ptrdiff_t *, double *, double *, ptrdiff_t *) = dgemv; 
    double (*dnrm2Ptr)(ptrdiff_t *, double *, ptrdiff_t *) = dnrm2;
#else
    void (*dgemvPtr)(char *, ptrdiff_t *, ptrdiff_t *, double *, double *, ptrdiff_t *, 
          double *, ptrdiff_t *, double *, double *, ptrdiff_t *) = dgemv_; 
    double (*dnrm2Ptr)(ptrdiff_t*, double *, ptrdiff_t *) = dnrm2; 
#endif

    /* Hack: if index != 1:k, we do MGS to avoid reshuffling */
    if ( *iflag == 1 ) {
        for ( i=0; i< *k; i++ ){
            if ( index[i] != (i+1) ) {
                *iflag = 0;
                break;
            }
        }
    }
    normv_old = 0;
    *nre = 0;  
    *normv = 0.0;

    while ( ( *normv < *alpha* normv_old) || ( *nre == 0 ) ) {
        if ( *iflag == 1 ) {
            /* CGS */
            dgemvPtr(&Transpose, &N, &K, &oneD,  V, &LDV, vnew, &one, &zero, work, &one); 
            dgemvPtr(&Normal,    &N, &K, &nOneD, V, &LDV, work, &one, &oneD, vnew, &one); 
        } else {
            /* MGS */
            MGS( n, k, V, ldv, vnew, index );
        }
        normv_old = *normv; 
        /* following line works! */
        *normv = dnrm2Ptr( &N, vnew, &one );    

        /* following line does not work: */
/*         *normv = dnrm2Ptr( (ptrdiff_t *)n, vnew, &one );    */

        *nre = *nre + 1;

        if ( *nre > MAXTRY ) {
            /* vnew is numerically in span(V) --> return vnew as all zeros */
            *normv = 0.0;
            for ( i=0; i< *n ; i++ )
                vnew[i] = 0.0;
            return;
        }

    }
}


