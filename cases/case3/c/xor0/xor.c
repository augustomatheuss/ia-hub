/* Xor Example with CMLP
 *
 * Copyright (c) 2016, Augusto Damasceno.
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

#include "../../mlp.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#ifdef __unix__
    #include <sys/time.h>
#endif

int main(int argc, char ** argv)
{
	/* Weights */
	double *** weights;
	
	/* History of MSE */	
	double * mse_history;

	/* Configuration of Training Data */
	training * trainingData = (training*) malloc(sizeof(training));
	trainingData->nlayers = 3;
	trainingData->neurons = (int *) \
	malloc(sizeof(int)*trainingData->nlayers);
	trainingData->neurons[0] = 3;
	trainingData->neurons[1] = 4;
	trainingData->neurons[2] = 1;
	trainingData->ninputs = 2;	
	trainingData->alpha = 1e-4;
	trainingData->bias = (double*) \
	malloc(sizeof(double)*trainingData->nlayers);
	trainingData->bias[0] = -1;
	trainingData->bias[1] = -1;
	trainingData->bias[2] = -1;
	trainingData->x = matrixAlloc(4,2);
	trainingData->x[0][0] = 0;
	trainingData->x[0][1] = 0;
	trainingData->x[1][0] = 0;
	trainingData->x[1][1] = 1;
	trainingData->x[2][0] = 1;
	trainingData->x[2][1] = 0;
	trainingData->x[3][0] = 1;
	trainingData->x[3][1] = 1;
	trainingData->examples = 4;
	trainingData->reference = matrixAlloc(4,1);
	trainingData->reference[0][0] = 0;
	trainingData->reference[1][0] = 1;
	trainingData->reference[2][0] = 1;
	trainingData->reference[3][0] = 0;
	trainingData->lrate = 0.375;
	trainingData->acceptedError = 1e-20;
	trainingData->maxIteration = 1e5;

	/* Allocation of weights */
	weights = initMLP(trainingData->neurons, \
	trainingData->nlayers,trainingData->ninputs);
	if(weights == NULL)
	{
#ifdef DEBUG_MODE
		printf("Memory error in initialization.\n"); 
#endif
		return 2;
	}

	printf("MLP LEARNING XOR LOGICAL OPERATION\n\n");

	/* Get time  */
#ifdef __unix__
    struct timeval t_begin,t_end;
    gettimeofday(&t_begin, NULL);
#else
    clock_t t = clock();
#endif

	/* MLP Training */
	mse_history = trainingMLP(weights,trainingData,"sigmoid");	
    
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

	/* Print MLP */
	printMLP(weights,trainingData);

	/* Status of XOR Learning */
	trainingPrint(trainingData);
	double * out;
	out = (double*) malloc(sizeof(double));
    printf("\nTraining Elapsed Time: %.6fs\n",elapsed);
	printf("Iterations: %ld\n", (long int) mse_history[0]*4);
	printf("Error: %.4e\n",mse_history[(int) mse_history[0]]);

	printf("XOR\n");
	printf("INPUT [0,0] : \t");
	outMLP(weights,trainingData,"sigmoid",trainingData->x,0,out);
	printf("%.8f\n",out[0]);
	printf("INPUT [0,1] : \t");
	outMLP(weights,trainingData,"sigmoid",trainingData->x,1,out);
	printf("%.8f\n",out[0]);
	printf("INPUT [1,0] : \t");
	outMLP(weights,trainingData,"sigmoid",trainingData->x,2,out);
	printf("%.8f\n",out[0]);
	printf("INPUT [1,1] : \t");
	outMLP(weights,trainingData,"sigmoid",trainingData->x,3,out);
	printf("%.8f\n",out[0]);	

	return 0;
}

