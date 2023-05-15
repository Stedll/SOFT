#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "fftw3.h"
#include "csecond.h"
#include "makeweights.h"
#include "so3_correlate_fftw.h"
#include "soft_fftw.h"

#include "s2_cospmls.h"
#include "s2_legendreTransforms.h"
#include "s2_semi_memo.h"

#define NORM( x ) ( (x[0])*(x[0]) + (x[1])*(x[1]) )

float* correlate (char * signalFile, char * patternFile, int bwIn, int bwOut, int degLim );

int main ( int argc,
	   char **argv )
{
  int bwIn, bwOut, degLim ;
  float* ret;
  if (argc < 6 )
    {

      printf("test_soft_fftw_correlate2 signalFile patternFile ");
      printf("bwIn bwOut degLim [result]\n");
      exit(0) ;
    }

  bwIn = atoi( argv[3] );
  bwOut = atoi( argv[4] );
  degLim = atoi( argv[5] );
  ret = correlate( argv[1], argv[2], bwIn, bwOut, degLim);
  printf(",%f,%f,%f,%f", //correlation,alpha,beta,gamma
   ret[3], ret[0], ret[1], ret[2]);
}

float* correlate (char * signalFile, char * patternFile, int bwIn, int bwOut, int degLim )
{
  fftw_init_threads();
  //printf("%s\n", signalFile);
  //printf("%s\n", patternFile);
  FILE *fp;
  int i ;
  int n ;
  double tstart, tstop ;
  fftw_complex *workspace1, *workspace2  ;
  double *workspace3 ;
  double *sigCoefR, *sigCoefI ;
  double *patCoefR, *patCoefI ;
  fftw_complex *so3Sig, *so3Coef ;
  fftw_plan_with_nthreads(24);
  //printf("%d", fftw_planner_nthreads());
  
  fftw_plan p1;
  int na[2], inembed[2], onembed[2] ;
  int rank, howmany, istride, idist, ostride, odist ;
  int tmp, maxloc, ii, jj, kk ;
  double maxval, tmpval ;

  n = 2 * bwIn ;

  so3Sig = fftw_malloc( sizeof(fftw_complex) * (8*bwOut*bwOut*bwOut) );
  workspace1 = fftw_malloc( sizeof(fftw_complex) * (8*bwOut*bwOut*bwOut) );
  workspace2 = fftw_malloc( sizeof(fftw_complex) * ((14*bwIn*bwIn) + (48 * bwIn)));
  workspace3 = (double *) malloc( sizeof(double) * (12*n + n*bwIn));
  sigCoefR = (double *) malloc( sizeof(double) * bwIn * bwIn ) ;
  sigCoefI = (double *) malloc( sizeof(double) * bwIn * bwIn ) ;
  patCoefR = (double *) malloc( sizeof(double) * bwIn * bwIn ) ;
  patCoefI = (double *) malloc( sizeof(double) * bwIn * bwIn ) ;
  so3Coef = fftw_malloc( sizeof(fftw_complex) * ((4*bwOut*bwOut*bwOut-bwOut)/3) ) ;

  /****
       At this point, check to see if all the memory has been
       allocated. If it has not, there's no point in going further.
  ****/

  if ( (so3Coef == NULL) ||
       (workspace1 == NULL) || (workspace2 == NULL) ||
       (workspace3 == NULL) ||
       (sigCoefR == NULL) || (sigCoefI == NULL) ||
       (patCoefR == NULL) || (patCoefI == NULL) ||
       (so3Sig == NULL) )
    {
      perror("Error in allocating memory");
      exit( 1 ) ;
    }


  /* create plan for inverse SO(3) transform */
  n = 2 * bwOut ;
  howmany = n*n ;
  idist = n ;
  odist = n ;
  rank = 2 ;
  inembed[0] = n ;
  inembed[1] = n*n ;
  onembed[0] = n ;
  onembed[1] = n*n ;
  istride = 1 ;
  ostride = 1 ;
  na[0] = 1 ;
  na[1] = n ;

  p1 = fftw_plan_many_dft( rank, na, howmany,
			   workspace1, inembed,
			   istride, idist,
			   so3Sig, onembed,
			   ostride, odist,
			   FFTW_FORWARD, FFTW_ESTIMATE );

  n = 2 * bwIn ;
  //printf("Reading in signal file\n");
  /* read in SIGNAL samples */
  /* first the signal */
  fp = fopen(signalFile,"r");
  for ( i = 0 ; i < bwIn * bwIn ; i ++ )
    {
      fscanf(fp,"%lf", sigCoefR + i);
      fscanf(fp,"%lf", sigCoefI + i);
    }
  fclose( fp );

  //printf("Reading in pattern file\n");
  /* read in SIGNAL samples */
  /* first the signal */
  fp = fopen(patternFile,"r");
  for ( i = 0 ; i < bwIn * bwIn ; i ++ )
    {
      fscanf(fp,"%lf", patCoefR + i);
      fscanf(fp,"%lf", patCoefI + i);
    }
  fclose( fp );

  //printf("about to combine coefficients\n");
  /* combine coefficients */
  tstart = csecond() ;
  so3CombineCoef_fftw( bwIn, bwOut, degLim,
		       sigCoefR, sigCoefI,
		       patCoefR, patCoefI,
		       so3Coef ) ;
  tstop = csecond();
  //fprintf(stderr,"combine time \t = %.4e\n", tstop - tstart);
  
  //printf("about to inverse so(3) transform\n");

  tstart = csecond();
  /* now inverse so(3) */
  Inverse_SO3_Naive_fftw(bwOut,
			  so3Coef,
			  so3Sig,
			  workspace1,
			  workspace2,
			  workspace3,
			  &p1,
			  1) ;
  tstop = csecond();

  /* now find max value */
  maxval = 0.0 ;
  maxloc = 0 ;
  for ( i = 0 ; i < 8*bwOut*bwOut*bwOut ; i ++ )
    {
      tmpval = NORM( so3Sig[i] );
      if ( tmpval > maxval )
      {
        maxval = tmpval;
        //printf("update:%f\n", maxval);
        maxloc = i;
      }
    }

    ii = floor(maxloc / (4. * bwOut * bwOut));
    tmp = -maxloc + (ii * 4. * bwOut * bwOut);
    jj = floor(tmp / (2. * bwOut));
    tmp = -maxloc + (ii * 4 * bwOut * bwOut) - jj * (2 * bwOut);
    kk = tmp;

    static float ret[4];
    ret[0] = M_PI * ((jj / ((double)bwOut)));
    ret[1] = M_PI * (2 * ii + 1) / (4. * bwOut);
    ret[2] = M_PI * kk / ((double)bwOut);
    ret[3] = maxval;

    fftw_destroy_plan(p1);

    fftw_free(so3Coef);
    free(patCoefI);
    free(patCoefR);
    free(sigCoefI);
    free(sigCoefR);
    free(workspace3);
    fftw_free(workspace2);
    fftw_free(workspace1);
    fftw_free(so3Sig);

    //printf("alpha = %f\nbeta = %f\ngamma = %f\ncorrelation = %f\n",
    // ret[0], ret[1], ret[2], ret[3]);
    return ret;
}
