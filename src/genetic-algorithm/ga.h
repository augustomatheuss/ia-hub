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

#ifndef _GA_H
#define _GA_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include <time.h>


/* Prints enabled/disabled */
#ifndef DEBUG_MODE
	#define DEBUG_MODE 0
#endif

/* GA Configuration, 1 to enable */

/* DATA TYPE */
/* WORKS WITH BINARY TYPES 8,16,32 and 64 BITS*/
#ifndef UINT_SZ
    #define UINT_SZ 64
#endif

/* DO NOT EDIT THIS BLOCK */
#if UINT_SZ == 64
    #define TYPE uint64_t
    #define TYPE_MAX 9223372036854775807.0
#elif UINT_SZ == 32
    #define TYPE uint32_t
    #define TYPE_MAX 4294967295.0
#elif UINT_SZ == 16
    #define TYPE uint16_t
    #define TYPE_MAX 65535.0
#elif UINT_SZ == 8
    #define TYPE uint8_t
    #define TYPE_MAX 255.0
#else
    #define TYPE_ERR
    #define UINT_SZ 32
    #define TYPE uint32_t
    #define TYPE_MAX 4294967295.0
#endif
/**************************/

/* POPULATION STRUCT */
typedef struct
{
    TYPE ** chromosomes;
    float * scores;
    int variables;
    int size;
    int elitSize;
	char * roulette_flags;
} population;

/* STOPPING CRITERION */
#ifndef STOP_GENERATIONS
    #define STOP_GENERATIONS 50
#endif

/* POPULATION  */
#ifndef POPULATION_INIT
    #define POPULATION_INIT 100
#endif
#ifndef POPULATION_CONST
    #define POPULATION_CONST 1
#endif
#ifndef POPULATION_MAX
    #define POPULATION_MAX 150
#endif

/* ELIT */
#ifndef ELIT_PERCENT
    #define ELIT_PERCENT 0.1
#endif

/* SELECTION TYPES */
#ifndef SELECTION_ROULETTE
    #define SELECTION_ROULETTE 1
#endif
#ifndef SELECTION_NORMAL
    #define SELECTION_NORMAL 0
#endif
#ifndef SELECTION_RANDOM
    #define SELECTION_RANDOM 0
#endif

/* CROSSOVER TYPES */
#ifndef CROSSOVER_ONE_POINT
    #define CROSSOVER_ONE_POINT 1
#endif
#ifndef CROSSOVER_TWO_POINT
    #define CROSSOVER_TWO_POINT 0
#endif

/* CROSSOVER CONFIGURATION  */
#ifndef CROSSOVER_RATE
    #define CROSSOVER_RATE 0.4
#endif

/* MUTATION TYPES */
#ifndef MUTATION_UNIFORM
    #define MUTATION_UNIFORM 1
#endif

/* MUTATION CONFIGURATION  */
#ifndef MUTATION_PROBABILITY
    #define MUTATION_PROBABILITY 0.015
#endif

/* Initialization */
int ga_init(population * p, int var);

/* Deallocate memory before exit */
void ga_end(population * p);

/* Normalize the scores and return the SUM */
float normalize(population * p);

/* Fitness */
void fitness(float(*func)(TYPE*), population * p);

/* Convert binary value to float in a range */
/* Based on quantized value */
float interpret(TYPE chromosome, float inf, float sup);

/* Crossover */
void crossover(population * p);

/* Selection */
/* Select a chromosome in the interval */
/* Return the the index for roulette */
/* Switch with begin position for normal and random */
int selection(population * p, int begin, int end);

/* Mutation */
void mutation(TYPE * chromosome, int var);

/* Print population */
void printPopulation(float(*func)(TYPE*),population * p);

/* Quick Sort, modified to keep the order of the chromosomes */
int compare(float a, float b);
void swapFloat(float * a, int pos1, int pos2);
void swapTypeMatrix(TYPE ** a, int y, int pos1, int pos2);
int partition(float * A, TYPE ** B, int y, int l, int r);
void quicksort(float * A, TYPE ** B, int y, int l, int r);
 
#endif /* _GA_H */

