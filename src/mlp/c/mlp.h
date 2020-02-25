/* C Multilayer Perceptron Neural Network Library
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

#ifndef _MLP_H
#define _MLP_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* Prints enabled/disabled */
#ifndef DEBUG_MODE
    #define DEBUG_MODE
#endif

/* Vector Rand Permutation */
void vperm(int * a, int sz);

/* Mean Square Error Batch Mode */
double mseb(double ** reference, double ** output,\
int nexamples, int noutputs);

/* Matrix Memory Allocation */
double ** matrixAlloc(int r, int c);

/* Get activation function id */
int getActv(char * name);

/* Sigmoid function */
void sigmoid(double ** ori, double ** dest, int layer, double sz);

/* Layer 0 Out  */
void layerOut0(double ** dest, double bias, \
double ** examples, int example, int ninputs, \
double *** weights, int nneurons, int activation);

/* Layer 1 and Following Out */
void layersOut(double ** dest, double bias, int nOutsPrev, \
double *** weights, int layer, int nneurons, int activation);

/* Error = (desired output) - (reference output) */
void errorMLP(double * error, double ** ref, int example, \
double ** yout, int layer, int outs);

/* Derivative of the activation function = df */
void dActivation(double ** df, double ** yout, \
int layer, int outs, int activation);

/* Local gradient of Last layer = Error * df. */
void gradientLast(double ** gradient, double * error, \
double ** df, int layer, int neurons);

/* Local gradient of Layer 0 until (layers-1) */
void gradient(double ** gradient, double ** wgs, \
double ** df, int layer, int neurons);

/* Copy Weights */
void weightsCopy(double *** ori, double *** dest,\
int * neurons, int nlayers, int ninputs);

/* Update Layer 1 and Following */
void updateLayer(double *** weights,double alpha, \
double *** weightsPast, double lrate, double ** gs, double bias, \
double ** yout, int layer, int neurons, int neuronsPrev);

/* Update Layer 0 */
void updateLayer0(double *** weights,double alpha, \
double *** weightsPast, double lrate, double ** gs, double bias, \
double ** ref, int example, int neurons, int ninputs);

/* SUM WtGs = sum of (next layer G * next layer weights) */
/* Ignore the weights relative to bias */
void sumWtGs(double ** wgs, double *** weights, double ** gs, \
int nextLayer, int neuronsNextLayer, int neurons);

/* Memory Allocation of the Weights */
double *** weightsAlloc(int * neurons, int nlayers, int ninputs);

/* Training data structure */
typedef struct
{
	int nlayers;
	int * neurons;
	int ninputs;	
	double alpha;
	double * bias;
	double ** x;
	int examples;
	double ** reference;
	double lrate;
	double acceptedError;
	long int maxIteration;
} training;

/* Deallocate memory of a traning struct */
void trainingDestruct(training * tr);

/* Print the training struct */
void trainingPrint(training * tr);

/* MLP initialization
 * weights = layers X neurons X weights
 * neurons = number of neurons by layer
 * nlayers = number of layers
 * ninputs = number of inputs
 * return weights 
 */
double *** initMLP(int * neurons, int nlayers, int ninputs);

/* MLP Training 
 * weights = layers X neurons X weights 
 * neurons = number of neurons by layer 
 * nlayers = number of layers 
 * ninputs = number of inputs 
 * alpha = momentum constant 
 * bias = value of bias by layer 
 * x = inputs
 * examples = size of training set 
 * ref = desired outputs 
 * lrate = learning-rate 
 * acceptedError = acceptable error 
 * maxIteration = maximum iteration 
 * activation = activation function 
 *   'sigmoid', ...
 * return History of MSE, the position 0 is the size of history
 */
double * trainingMLP(double *** weights, training * trainingData, \
char * activation);

/* Output of MLP */
/* The 'in' is a matrix of inputs X 'pos' */
void outMLP(double *** weights, training * trainingData, \
char * activation, double ** in, int pos, double * out);

/* Save training data and weights of MLP in a file */
void saveMLP(char * filename, training * trainingData, \
double *** weights);

/* Load the examples, references and configuration from files.
 * 
 * The format for the inputs file is a matrix examples X inputs, 
 *   elements separated by space and lines by break line.
 *
 * The format for the outputs file is a matrix examples X outputs, 
 *   elements separated by space and lines by break line.
 *
 * The format of configuration file is: 
 *   Line 0: Number of Layers 
 *   Line 1: Number of neurons for each layer separated by space 
 *   Line 2: Number of inputs 
 *   Line 3: Bias separated by space 
 *   Line 4: Alpha 
 *   Line 5: Learning rate 
 *   Line 6: Accepted Error 
 *   Line 7: Maximum Iteration 
 *
 * The files inputs and outputs must follow the configuration.
 *   Error in inputs return -1
 *   Error in outputs return -2
 *   Error in conf return -3
 */
int loadExamplesFromFile(char * inputs, char * outputs, \
char * conf, training * trainingData);

/* Print the neural network */
void printMLP(double *** weights, training * trainingData);

#endif /* _MLP_H */
