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

#ifdef TYPE_ERR
#ifdef DEBUG_MODE
    printf("WARNING. UINT_SZ INCORRECT, 32 ENABLED BY DEFAULT\n");
#endif
#endif

	/* Create initial population, 'VAR' variables */
    population popul;
    if(ga_init(&popul,VAR))
    {
#ifdef DEBUG_MODE
        printf("Memory error in initialization.\n");
#endif
        return 2;
    }

#ifdef DEBUG_MODE
    printf("GA initialized.\n" \
        "Population: %d\nElit: %d\n" \
        "Stopping criterion: %d generations\n\n", \
        popul.size,popul.elitSize,STOP_GENERATIONS);
#endif

    /* Get time  */
#ifdef __unix__
    struct timeval t_begin,t_end;
    gettimeofday(&t_begin, NULL);
#else
    clock_t t = clock();
#endif

    int generations = 0;
    while(generations < STOP_GENERATIONS)
    {
        /* Fitness */
        fitness(my_fitness,&popul,thread_count);
        
        /* Crossover  */
        crossover(&popul);
 
        generations++;
    }

    /* Get time  */
    float elapsed;
#ifdef __unix__
    gettimeofday(&t_end, NULL);
    elapsed = ((t_end.tv_sec+t_end.tv_usec/1000000.0)) - \
            (t_begin.tv_sec+t_begin.tv_usec/1000000.0); 
#else
    t = clock() - t;
    elapsed = ((float)t)/CLOCKS_PER_SEC;
#endif   

    /* Apply fitness to sort, and print last population */
    fitness(my_fitness,&popul,thread_count);
    ///printPopulation(my_fitness,&popul);

    /* Print best solution */
    printf("\nGeneration: %d.\nBest Solution:\n",generations);
    printf("x1: %f , x2: %f\n", \
        interpret(popul.chromosomes[popul.size-1][0],\
        LIMIT_INFERIOR_X1,LIMIT_SUPERIOR_X1),\
        interpret(popul.chromosomes[popul.size-1][1],\
        LIMIT_INFERIOR_X2,LIMIT_SUPERIOR_X2));
    printf("y = %f\n",\
        my_fitness(popul.chromosomes[popul.size-1]));
   
    /* Print elapsed time */
    printf("\nElapsed time: %.6fs\n",elapsed); 

    /* Free memory allocated */
    ga_end(&popul);

    return 0;
} 

