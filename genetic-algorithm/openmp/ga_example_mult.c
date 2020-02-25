/* This file is part of gal software */

#include <omp.h>
#include "ga.h"
#include <float.h>
#ifdef __unix__
    #include <sys/time.h>
#endif

#define LIMIT_INFERIOR_X1 -100.0
#define LIMIT_SUPERIOR_X1 100.0
#define LIMIT_INFERIOR_X2 -100.0
#define LIMIT_SUPERIOR_X2 100.0
#define VAR 2

#define TESTS_TO_DO 30
/* You should disable DEBUG_MODE in ga.h */

/* Optimize Schaffer F6 function */
/* Min in x1 = 0.0 and x2 = 0.0 */
float my_fitness(TYPE * c)
{
    float x1 = \
        interpret(c[0],LIMIT_INFERIOR_X1,LIMIT_SUPERIOR_X1);
    float x2 = \
        interpret(c[1],LIMIT_INFERIOR_X2,LIMIT_SUPERIOR_X2);

	float xpow2 = x1*x1;
	float ypow2 = x2*x2;
	float up = sin(sqrt(xpow2 + ypow2));
	float down = 1.0 + 0.001 * (xpow2 + ypow2);

	return 0.5 + (up*up - 0.5) / (down * down);
}

int main(int argc, char ** argv)
{
	/*OpenMP*/
	int thread_count = strtol(argv[1],NULL,10);

	struct timeval t_begin,t_end;
	int i;
	int j;
	float elapsed[TESTS_TO_DO];
	float best = 1000000;
	float best_var[VAR];
	float worst = -1;
	float worst_var[VAR];
	float solution;
	int generations;
	population popul;
	for(i=0;i<TESTS_TO_DO;i++)
	{
		ga_init(&popul,VAR);
    
		/* Get time  */
#ifdef __unix__
		gettimeofday(&t_begin, NULL);
#else
		clock_t t = clock();
#endif

		generations = 0;
		while(generations < STOP_GENERATIONS)
		{
			/* Fitness */
			fitness(my_fitness,&popul,thread_count);
        
			/* Crossover  */
			crossover(&popul,thread_count);
 
			generations++;
		}

		/* Get time  */
#ifdef __unix__
		gettimeofday(&t_end, NULL);
		elapsed[i] = ((t_end.tv_sec+t_end.tv_usec/1000000.0)) - \
        (t_begin.tv_sec+t_begin.tv_usec/1000000.0); 
#else
		t = clock() - t;
		elapsed[i] = ((float)t)/CLOCKS_PER_SEC;
#endif   

		/* Apply fitness to sort */
		fitness(my_fitness,&popul,thread_count);

		solution = my_fitness(popul.chromosomes[popul.size-1]);
		if(solution < best)
		{
			best = solution;
			for(j=0;j<VAR;j++)
			{
				best_var[j] = \
				interpret(popul.chromosomes[popul.size-1][j], \
				LIMIT_INFERIOR_X1,LIMIT_SUPERIOR_X1);
			}
		}
		else if (solution > worst)
		{
			worst = solution;
			for(j=0;j<VAR;j++)
			{
				worst_var[j] = \
				interpret(popul.chromosomes[popul.size-1][j], \
				LIMIT_INFERIOR_X1,LIMIT_SUPERIOR_X1);
			}
		}

		/* Free memory allocated */
		ga_end(&popul);
	}	
   
	/* Print results */
	float mean = 0.0;
	for(i=0;i<TESTS_TO_DO;i++)
	{
		printf("Elapsed Time %i: %.6fs\n",i,elapsed[i]);
		mean += elapsed[i];	 
	}
	printf("Elapsed Time - Mean: %f\n\n",(mean/TESTS_TO_DO));

	printf("Worst Result:\t%.8f\n",worst);
	for(j=0;j<VAR;j++)
		printf("x%d: %.8f\n",j,worst_var[j]);

	printf("Best Result:\t%.8f\n",best);
	for(j=0;j<VAR;j++)
		printf("x%d: %.8f\n",j,best_var[j]);

       return 0;
} 

