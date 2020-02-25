/* Genetic Algorithm Lightweight Implementations
 *
 * Copyright (c) 2015,2016, Augusto Damasceno.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ga.h"

int ga_init(population * p, int var)
{
#if POPULATION_CONST == 1
    p->chromosomes = (TYPE **) \
        malloc(sizeof(TYPE*)*POPULATION_INIT);
    if(p->chromosomes == NULL)
            return 1;
    
    int i;
    for(i=0;i<POPULATION_INIT;i++)
    { 
        p->chromosomes[i] = (TYPE *) \
            malloc(sizeof(TYPE)*var);
        if(p->chromosomes[i] == NULL)
            return 1;
    }

    p->scores = (float *) \
        malloc(sizeof(float)*POPULATION_INIT);
    if(p->scores == NULL)
        return 1;

	p->roulette_flags = (char *) \
        malloc(sizeof(char)*POPULATION_INIT);
    if(p->roulette_flags == NULL)
        return 1;

#else
    p->chromosomes = (TYPE **) \
        malloc(sizeof(TYPE*)*POPULATION_MAX);
    if(p->chromosomes == NULL)
            return 1;
    
    int i;
    for(i=0;i<POPULATION_MAX;i++)
    {
        p->chromosomes[i] = (TYPE *) \
            malloc(sizeof(TYPE)*var);
        if(p->chromosomes[i] == NULL)
            return 1;
    }

    p->scores = (float *) \
        malloc(sizeof(float)*POPULATION_MAX);
    if(p->scores == NULL)
        return 1;

	p->roulette_flags = (char *) \
        malloc(sizeof(char)*POPULATION_MAX);
    if(p->roulette_flags == NULL)
        return 1;

#endif 
    p->variables = var;
    p->size = POPULATION_INIT;
    p->elitSize = ceil(POPULATION_INIT*ELIT_PERCENT);

    int v;
    srand (time(NULL));
#if DEBUG_MODE == 1
    for(i=0; i < p->size; i++)
    {
        for(v=0;v<var;v++)
        {
            p->chromosomes[i][v] = rand();
            printf("Chromosome %d - x%d: %u\n", \
                i,v+1,p->chromosomes[i][v]);
        }
    }
    printf("\n");
#else
    for(i=0; i < p->size; i++)
        for(v=0;v<var;v++)
            p->chromosomes[i][v] = rand();
#endif

    return 0;
}   

void ga_end(population * p)
{
    int i;
#if POPULATION_CONST == 1 
    for(i=0; i < POPULATION_INIT; i++)
#else
    for(i=0; i < POPULATION_MAX; i++)
#endif
    free(p->chromosomes[i]);

    free(p->scores);
}

float normalize(population * p)
{
	int i;
	float offset = 0;

	for(i=0;i<p->size;i++)
	{
		if (p->scores[i] < offset)
			offset = p->scores[i];
	}
	offset *= -1;
	p->scores[0] += offset;
	float max = p->scores[0];
	for(i=1;i<p->size;i++)
	{
		p->scores[i] += offset;
		if(p->scores[i] > max)
			max = p->scores[i];
	}

	float sum = 0;
	for(i=0; i < p->size; i++)
	{
		p->scores[i] /= max;
		sum += p->scores[i];
	}
	
	return sum;
}

void fitness(float(*func)(TYPE*),population * p)
{
    int i;
    for(i=0;i<p->size;i++) 
        p->scores[i] = -1 * func(p->chromosomes[i]);
 
    quicksort(p->scores,p->chromosomes,p->variables,0,p->size-1);
}

float quantized(TYPE chromosome)
{
    float sum = 0.0;
	int i;
	TYPE getbit = 1;
	for(i=1;i<=UINT_SZ;i++)
	{
		if (chromosome & getbit)
			sum += pow(2.0,(double)(-1*i));

		getbit = getbit<<1;
	}
	sum += pow(2.0,(double) (-1*(UINT_SZ+1)));

	return sum; 
}

float interpret(TYPE chromosome, float inf, float sup)
{
	return (quantized(chromosome)*(sup-inf)+inf);
}

int selection(population * p, int begin, int end)
{
#if SELECTION_RANDOM == 1

    swapTypeMatrix( p->chromosomes , p->variables, begin , \
        ( (rand() % (end+1-begin)) + begin ) );

	return 0;

#elif SELECTION_NORMAL == 1

    float x1 = (float) rand()/RAND_MAX;
    float x2 = (float) rand()/RAND_MAX;
    float s = x1*x1+x2*x2;
    while (s >= 1 || s == 0)
    {
        x1 = (float) rand()/RAND_MAX; 
        x2 = (float) rand()/RAND_MAX;    
        s = x1*x1+x2*x2;
    }
    
    float randnorm = x1 * sqrt( -2.0 * log(s) / s ); 
    int sel = (end+1)-ceil( (randnorm /4) * (end-begin+1) );
    swapTypeMatrix(p->chromosomes,p->variables,begin,sel);

	return 0;

#elif SELECTION_ROULETTE == 1

	float roulette;
	int i;
	int sel = -1;

	while ((p->roulette_flags[sel] != 0) || (sel < 0))
	{
		roulette = ((float) rand())/ ((float)RAND_MAX);

		for(i=end;i>begin;i--) 
		{
			sel = i;
			if( (roulette > p->scores[sel-1]) && \
			(roulette <= p->scores[sel]) )
				break;
		}
		if( roulette <= p->scores[1] )
			sel = 0;
	}
	
	if ( roulette <= p->scores[1] )
	{
		p->roulette_flags[0] = 1;
		return 0;
	}
	else
	{
		p->roulette_flags[sel] = 1;
		return sel;
	}

#endif
}

void crossover(population * p)
{
#if UINT_SZ == 64
    #if CROSSOVER_ONE_POINT == 1
        uint64_t maskA = 0xFFFFFFFF00000000;
        uint64_t maskB = 0x00000000FFFFFFFF;
    #elif CROSSOVER_TWO_POINT == 1
        uint64_t maskA = 0xFFFF0000FFFF0000;
        uint64_t maskB = 0x0000FFFF0000FFFF;
    #endif
#elif UINT_SZ == 32
    #if CROSSOVER_ONE_POINT == 1
        uint32_t maskA = 0xFFFF0000;
        uint32_t maskB = 0x0000FFFF;
    #elif CROSSOVER_TWO_POINT == 1
        uint32_t maskA = 0xFF00FF00;
        uint32_t maskB = 0x00FF00FF;
    #endif
#elif UINT_SZ == 16
    #if CROSSOVER_ONE_POINT == 1
        uint16_t maskA = 0xFF00;
        uint16_t maskB = 0x00FF;
    #elif CROSSOVER_TWO_POINT == 1
        uint16_t maskA = 0xF0F0;
        uint16_t maskB = 0x0F0F;
    #endif
#elif UINT_SZ == 8
    #if CROSSOVER_ONE_POINT == 1
        uint8_t maskA = 0xF0;
        uint8_t maskB = 0x0F;
    #elif CROSSOVER_TWO_POINT == 1
        uint8_t maskA = 0b11001100;
        uint8_t maskB = 0b00110011;
    #endif
#endif

#if SELECTION_ROULETTE == 1
	float totalScore = normalize(p);
	
	int j;
	float sum = 0.0;
	for(j=0;j<p->size;j++)
	{
		p->scores[j] = sum + p->scores[j]/totalScore;
		sum = p->scores[j];
		p->roulette_flags[j] = 0;
	}
#endif

    int selA;
    int selB;
    TYPE * childrenA = \
        (TYPE*) malloc(sizeof(TYPE)*p->variables);
    TYPE * childrenB = \
        (TYPE*) malloc(sizeof(TYPE)*p->variables);   

    int i,v;
    int sizeAdd;
    int total = (int) ( (p->size - p->elitSize) * CROSSOVER_RATE);
	int end = p->size - p->elitSize;
    for(i=0; i < total/2; i++) 
    {

#if SELECTION_ROULETTE == 1
		selA = selection(p,0,end);
		selB = selection(p,0,end);
#else
        selA = i*2;
        selB = selA+1;
        selection(p,selA,end);
        selection(p,selB,end);
#endif
        for(v=0;v < p->variables; v++)
        {
            childrenA[v] = (p->chromosomes[selA][v] & maskA) | \
				(p->chromosomes[selB][v] & maskB);
            childrenB[v] = (p->chromosomes[selB][v] & maskA) | \
                (p->chromosomes[selA][v] & maskB);
        }
		mutation(childrenA,p->variables);
        mutation(childrenB,p->variables);
#if POPULATION_CONST == 1
        for(v=0;v < p->variables; v++)
        {
            p->chromosomes[selA][v] = childrenA[v];
            p->chromosomes[selB][v] = childrenB[v];
        }
#else
        sizeAdd = 0;
        if(p->size < POPULATION_MAX-1)
        {
            p->size += 2;
            sizeAdd = 1;
        }

        for(v=0;v < p->variables; v++)
        {   
            if(sizeAdd)
            {
                p->chromosomes[p->size-2][v] = childrenA[v];
                p->chromosomes[p->size-1][v] = childrenB[v];
                sizeAdd = 0;
            }
            else
            {
                p->chromosomes[selA][v] = childrenA[v];
                p->chromosomes[selB][v] = childrenB[v];
            }
        }                
#endif 
	}
}

void mutation(TYPE * chromosome, int var)
{
    int v;
    TYPE mask = 1;
    float rnd = ( ((float)rand())/((float) RAND_MAX));

	for(v=0; v<var; v++)
    {
		if(rnd <= (MUTATION_PROBABILITY/var) )
		{
			mask = mask << ( rand() % UINT_SZ );

			if(chromosome[v] & mask)
			{
				chromosome[v] = chromosome[v] & (~mask);   
			}
			else
			{
				chromosome[v] = chromosome[v] | mask;
			}
		}
	}
}

void printPopulation(float(*func)(TYPE*),population * p)
{
    int i,v;
    for(i=0;i<p->size;i++)
    {
        printf("Chromosome %d\n",i);
        for(v=0; v < p->variables; v++)
        {
            printf("x%d: %u\n",v+1,p->chromosomes[i][v]);
        }
        printf("Score: %f \n", \
            func(p->chromosomes[i]));
    }
}
    
int compare(float a, float b)
{
    if (a < b)
        return -1;
    if (a > b)
        return 1;
    return 0; 
}

void swapFloat(float * a, int pos1, int pos2)
{
    float sw = a[pos1];
    a[pos1] = a[pos2];
    a[pos2] = sw;    
}

void swapTypeMatrix(TYPE ** a, int y, int pos1, int pos2)
{
    int j;
    TYPE sw;
    for (j=0;j<y;j++)
    {
        sw = a[pos1][j];
        a[pos1][j] = a[pos2][j];
        a[pos2][j] = sw;
    }    
}

int partition(float* A, TYPE ** B, int y, int l, int r)
{                                      
    int i = l-1;
    int j;                  
    for(j=l; j<r; j++)    
    {                              
        if(compare(A[j],A[r]) != 1)    
        {                      
            i++;
            swapFloat(A,i,j);
            swapTypeMatrix(B,y,i,j);         
        }                   
    }
    swapFloat(A,i+1,r);
    swapTypeMatrix(B,y,i+1,r);
       
    return (i+1);
}

void quicksort(float * A,TYPE ** B, int y, int l, int r)
{                      
    int p;              
    if (l < r)                  
    {                  
        p = partition(A,B,y,l,r);       
        quicksort(A,B,y,l,p-1);    
        quicksort(A,B,y,p+1, r);  
    }                  
}

