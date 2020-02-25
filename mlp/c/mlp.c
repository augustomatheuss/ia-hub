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

#include "mlp.h"

/* Integer Vector Rand Permutation */
/* The pseudo-random number generator must be initialized */
void vperm(int * a, int sz)
{
	int i;
	int m = ceil(sz/2);
	int sw;
	int r;
	for(i=0; i<m; i++)
	{
		r = rand() % sz;
		sw = a[i];
		a[i] = a[r];
		a[r] = sw;
	}
}

/* Mean Square Error Batch Mode */
double mseb(double ** reference, double ** output,\
int nexamples, int noutputs)
{
	int e;
	int o;
	double serror;
	double sum;
	double mseb = 0.0;
	for(e=0; e<nexamples; e++)
	{
		sum = 0.0;
		for(o=0; o<noutputs; o++)
		{
			serror = (reference[e][o] - output[e][o]);
			serror *= serror;
			sum += serror;
		}
		mseb += sum/noutputs;
	}
	
	return (mseb/nexamples);
}

/* Matrix Memory Allocation */
double ** matrixAlloc(int r, int c)
{
	double ** m = (double**) malloc(sizeof(double*)*r);
	int i;
	for(i=0; i<r; i++)
		m[i] = (double*) malloc(sizeof(double)*c);
	
	return m;
}

/* Get activation function id */
int getActv(char * name)
{
	if (strcmp(name,"sigmoid") == 0)
	{
		return 1;
	}
	/* Reserved to update */
	else if (strcmp(name,"activation2") == 0)
	{
		return 1;
	}
	else
	{
		return 1;
	}
}

/* Sigmoid Function */
void sigmoid(double ** ori, double ** dest, int layer,  double sz)
{
	int i;
	for(i=0; i<sz; i++)
		dest[layer][i]= 1.0/(1.0+exp(-1.0*ori[layer][i]));
}

/* Layer 0 Out */
void layerOut0(double ** dest, double bias, \
double ** examples, int example, int ninputs, \
double *** weights, int nneurons, int activation)
{
	int i;
	int j;
	double sum;
	
	for(i=0; i<(nneurons); i++)
	{
		sum = 0;
		for(j=0; j<(ninputs+1); j++)
		{
			if (j)
			{
				sum += examples[example][j-1] * weights[0][i][j];
			}
			else
			{
				sum += bias * weights[0][i][j];
			}
		}
		dest[0][i] = sum;
	}	

	switch(activation)
	{	
		case 1:
			sigmoid(dest,dest,0,nneurons);
			break;
		default:
			sigmoid(dest,dest,0,nneurons);
			break;
	}	
}

/* Layer 1 and Following Out */
void layersOut(double ** dest, double bias, int nOutsPrev, \
double *** weights, int layer, int nneurons, int activation)
{
	int i;
	int j;
	double sum;
	
	for(i=0; i<(nneurons); i++)
	{
		sum = 0;
		for(j=0; j<(nOutsPrev+1); j++)
		{
			if (j)
			{
				sum += dest[layer-1][j-1]*weights[layer][i][j];
			}
			else
			{
				sum += bias*weights[layer][i][j];
			}
		}
		dest[layer][i] = sum;
	}	

	switch(activation)
	{	
		case 1:
			sigmoid(dest,dest,layer,nneurons);
			break;
		default:
			sigmoid(dest,dest,layer,nneurons);
			break;
	}	
}

/* Error = (desired output) - (reference output) */
void errorMLP(double *error, double ** ref, int example, \
double ** yout, int layer, int outs)
{
	int i;
	for(i=0; i<outs; i++)
		error[i] = ref[example][i] - yout[layer][i];
}

/* Derivative of the activation function = df */
void dActivation(double ** df, double ** yout, \
int layer, int outs, int activation)
{
	int i;
	switch(activation)
	{
		case 1:
			for(i=0; i<outs; i++)
				df[layer][i] = yout[layer][i] * (1 - yout[layer][i]);
			break;
		default:
			for(i=0; i<outs; i++)
				df[layer][i] = yout[layer][i] * (1 - yout[layer][i]);
			break;
	}
}

/* Local gradient. Last layer = Error * df.	*/
void gradientLast(double ** gradient, double * error, \
double ** df, int layer, int neurons)
{
	int i;
	for(i=0; i<neurons; i++)
		gradient[layer][i] = error[i] * df[layer][i];
}

/* Local gradient. Layer 0 until (layers-1) */
void gradient(double ** gradient, double ** wgs, \
double ** df, int layer, int neurons)
{
	int i;
	for(i=0; i<neurons; i++)
		gradient[layer][i] = wgs[layer][i] * df[layer][i];
}

/* Copy Weights */
void weightsCopy(double *** ori, double *** dest,\
int * neurons, int nlayers, int ninputs)
{
	double *** weights = (double***) \
	weightsAlloc(neurons, nlayers, ninputs);
	
	int layer;
	int neuron;
	int weight;	
	for(layer=0; layer<nlayers; layer++)
	{
		for(neuron=0; neuron<neurons[layer]; neuron++)
		{
			if(layer)
			{
				for(weight=0; weight<(neurons[layer-1]+1); weight++)
				{
					dest[layer][neuron][weight] =\
					ori[layer][neuron][weight];	
				}
			}
			else
			{
				for(weight=0; weight<(ninputs+1); weight++)
				{
					dest[layer][neuron][weight] =\
					ori[layer][neuron][weight];	
				}
			}
		}
	}
}

/* Update Layer 1 and Following */
void updateLayer(double *** weights,double alpha, \
double *** weightsPast, double lrate, double ** gs, double bias, \
double ** yout, int layer, int neurons, int neuronsPrev)
{
	int neuron;
	int weight;
	for(neuron=0; neuron<neurons; neuron++)
	{
		for(weight=0; weight<(neuronsPrev+1); weight++)
		{
			if(weight)
			{
				weights[layer][neuron][weight] = \
				weights[layer][neuron][weight] + \
				alpha * weightsPast[layer][neuron][weight] + \
				lrate*gs[layer][neuron]*yout[layer-1][weight-1];
			}
			else
			{
				weights[layer][neuron][weight] = \
				weights[layer][neuron][weight] + \
				alpha * weightsPast[layer][neuron][weight] + \
				lrate*gs[layer][neuron]*bias;	
			}
		}
	}	
}

/* Update layer 0 */
void updateLayer0(double *** weights,double alpha, \
double *** weightsPast, double lrate, double ** gs, double bias, \
double ** ref, int example, int neurons, int ninputs)
{
	int neuron;
	int weight;
	for(neuron=0; neuron<neurons; neuron++)
	{
		for(weight=0; weight<(ninputs+1); weight++)
		{
			if(weight)
			{
				weights[0][neuron][weight] = \
				weights[0][neuron][weight] + \
				alpha * weightsPast[0][neuron][weight] + \
				lrate*gs[0][neuron]*ref[example][weight-1];
			}
			else
			{
				weights[0][neuron][weight] = \
				weights[0][neuron][weight] + \
				alpha * weightsPast[0][neuron][weight] + \
				lrate*gs[0][neuron]*bias;	
			}
		}
	}	
}
/* SUM WtGs = sum of (next layer G * next layer weights) */
/* Ignore the weights relative to bias */
void sumWtGs(double ** wgs, double *** weights, double ** gs, \
int nextLayer, int neuronsNextLayer, int neurons)
{
	int neuron;
	int weight;
	for(weight=1; weight<(neurons+1); weight++)
	{
		wgs[nextLayer-1][weight] = 0; 
		for(neuron=0; neuron<(neuronsNextLayer); neuron++)
		{
			wgs[nextLayer-1][weight-1] += gs[nextLayer][weight] * \
			weights[nextLayer][neuron][weight];
		}
	}
}

/* Memory Allocation of the Weights */
double *** weightsAlloc(int * neurons, int nlayers, int ninputs)
{
	int layer;
	int neuron;
	int weight;
	
	double *** weights =\
	(double ***) malloc(sizeof(double**)*nlayers); 
	if(weights == NULL)
		return NULL;

	for(layer=0; layer<nlayers; layer++)
    {
		weights[layer] = \
		(double **) malloc(sizeof(double*)*neurons[layer]); 
		if(weights[layer] == NULL)
			return NULL;

		for(neuron=0; neuron<neurons[layer]; neuron++)
		{
			if(layer)
			{
				weights[layer][neuron] = (double *) \
				malloc(sizeof(double)*(neurons[layer-1]+1));
			}
			else
			{
				weights[layer][neuron] = (double *) \
				malloc(sizeof(double)*(ninputs+1));
				
			}
			if(weights[layer][neuron] == NULL)
				return NULL;
		}
    }
	
	return weights;
}

/* Deallocate memory of a traning struct */
void trainingDestruct(training * tr)
{	
	int i;
	free(tr->neurons);
	free(tr->bias);
	for(i=1; i<tr->examples; i++)
	{
		free(tr->x[i]);
		free(tr->reference[i]);
	}
	free(tr->x);
	free(tr->reference);	
	free(tr);
}

/* Print the training struct */
void trainingPrint(training * tr)
{
	int i;
	printf("Training Data\n\n");
	printf(">> Architecture\n");
	printf("Layers: %d\n",tr->nlayers);
	for(i=0; i<tr->nlayers; i++)
	{
		if(tr->neurons[i] == 1)
		{
			printf("\tLayer %d: 1 neuron\n",i);
		}
		else
		{
			printf("\tLayer %d: %d neurons\n",i,tr->neurons[i]);
		}
	}

	for(i=0; i<tr->nlayers; i++)
		printf("\tBias of Layer %d: %f\n",i,tr->bias[i]);

	printf("Inputs: %d\n",tr->ninputs);
	printf("\n>> Training Parameters\n");
	printf("Alpha: %.4e\n",tr->alpha);
	printf("Learning Rate: %.8f\n",tr->lrate);
	printf("Accepted Error: %.4e\n",tr->acceptedError);
	printf("Maximum Iteration: %ld\n",tr->maxIteration);
}

/* MLP initialization */
double *** initMLP(int * neurons, int nlayers, int ninputs)
{
#ifdef DEBUG_MODE
	printf("Initializing MLP weights\n");
#endif

	double *** weights = weightsAlloc(neurons, nlayers, ninputs);
	
	int layer;
	int neuron;
	int weight;	
	srand(time(NULL));
	for(layer=0; layer<nlayers; layer++)
	{
		for(neuron=0; neuron<neurons[layer]; neuron++)
		{
			if(layer)
			{
				for(weight=0; weight<(neurons[layer-1]+1); weight++)
				{
					weights[layer][neuron][weight] =\
					(double) rand()/RAND_MAX;
					#ifdef DEBUG_MODE
						printf("Layer %d - Neuron %d",layer,neuron);
						printf(" - Weight %d: %.6f\n",\
						weight,weights[layer][neuron][weight]);
					#endif
				}
			}
			else
			{
				for(weight=0; weight<(ninputs+1); weight++)
				{
					weights[layer][neuron][weight] =\
					(double) rand()/RAND_MAX;
					#ifdef DEBUG_MODE
						printf("Layer %d - Neuron %d",layer,neuron);
						printf(" - Weight %d: %.6f\n",\
						weight,weights[layer][neuron][weight]);
					#endif
				}
			}
		}
	}
	#ifdef DEBUG_MODE
		printf("\n");
	#endif

	return weights;
}

/* MLP Training */
double * trainingMLP(double *** weights, training * trainingData, \
char * activation)
{
	int nlayers = trainingData->nlayers;
	int * neurons = trainingData->neurons;
	int ninputs = trainingData->ninputs;
	double alpha = trainingData->alpha;
	double * bias = trainingData->bias;
	double ** x = trainingData->x;
	int examples = trainingData->examples;
	double ** ref = trainingData->reference;
	double lrate = trainingData->lrate;
	double acceptedError = trainingData->acceptedError;
	long int maxIteration = trainingData->maxIteration;

	int layer;
	int neuron;
	int weight;
	int i;
	int j;
	int actv = getActv(activation);

	/* Memory of past weights and swap weights */
	double *** weightsPast = weightsAlloc(neurons, nlayers, ninputs);
	double *** weightsSwap = weightsAlloc(neurons, nlayers, ninputs);
	for(layer=0; layer<nlayers; layer++)
	{
		for(neuron=0; neuron<neurons[layer]; neuron++)
		{
			if(layer)
			{
				for(weight=0; weight<(neurons[layer-1]+1); weight++)
					weightsPast[layer][neuron][weight] = 0;
			}
			else
			{
				for(weight=0; weight<(ninputs+1); weight++)
					weightsPast[layer][neuron][weight] = 0;
			}
		}
	}	

	if(weightsPast == NULL)
		return NULL;

	/* Memory for layers outputs */
	double ** yout = (double**) malloc(sizeof(double*)*nlayers);
	for(i=0; i<nlayers; i++)
		yout[i] = (double*) malloc(sizeof(double)*neurons[i]);
		
	/* Memory for last layers outputs of the examples */
	double ** youtLastLayers = (double**) \
	malloc(sizeof(double*)*examples);
	for(i=0; i<examples; i++)
	{
		youtLastLayers[i] = (double*) \
		malloc(sizeof(double)*neurons[nlayers-1]);
	}

	/* MLP error */
	double * error = (double*) \
	malloc(sizeof(double)*neurons[nlayers-1]);

	/* Derivative of the activation function */
	double ** df = (double**) \
	malloc(sizeof(double*)*nlayers);
	for(i=0; i<nlayers; i++)
		df[i] = (double*) malloc(sizeof(double)*neurons[i]);

	/* Local gradient. */
	/* Last layer = Error * df.	*/
	/* Others layers = df * SUM */
	/* SUM = sum of (next layer G * next layer weights) */
	double ** gs = (double**) \
	malloc(sizeof(double*)*nlayers);
	for(i=0; i<nlayers; i++)
		gs[i] = (double*) malloc(sizeof(double)*neurons[i]);
	
	/* SUM WtGs = sum of (next layer G * next layer weights) */
	/* Ignore the weights relative to bias */
	double ** wgs = (double**) \
	malloc(sizeof(double*)*nlayers-1);
	for(i=0; i<(nlayers-1); i++)
		wgs[i] = (double*) malloc(sizeof(double)*neurons[i]);

	/* Mean Square Error */
	double mse = acceptedError+1;

	/* Index of examples */
	int * xidx = (int*) malloc(sizeof(int)*examples);
	for(i=0; i<examples; i++)
		xidx[i] = i;
	
	/* Training loop */
	int ex = 0;
	long int counter = 0;
	/* The position 0 of mse_hist is the last position of history */
	long int mse_counter = 1;
	double * mse_hist = (double*) \
	malloc(sizeof(double)*floor(maxIteration/examples));
	int progress = 0;
	int displayStep = ceil(0.05*maxIteration);
	vperm(xidx,examples);
	while(mse > acceptedError && counter < maxIteration)
	{
		/* Iteration Number */
		counter++;
		
		/* Number of Training Example */
		ex = (ex % examples)+1;

		/* Propagation */	
		
		/* Out of first layer */
		layerOut0(yout,bias[0],x,xidx[(ex-1)],ninputs,\
		weights,neurons[0],actv);

		for(layer=1; layer<nlayers; layer++)
		{
			/* Out of "layer" layer */
			layersOut(yout,bias[layer],neurons[layer-1], \
			weights,layer,neurons[layer],actv);
		}

		/* Backpropagation */
		
		/* Neuron Update = learning-rate*G*y(layer-1) */

		/* Error = yref - y */
		errorMLP(error,ref,xidx[(ex-1)],yout,nlayers-1, \
		neurons[nlayers-1]);
		
		/* Derivative of the activation function = df */
		dActivation(df,yout,nlayers-1,neurons[nlayers-1],actv);

		/* Local gradient. Last layer = Error * df	*/
		gradientLast(gs,error,df,nlayers-1,neurons[nlayers-1]);

		/* Save weights in swap weights*/
		weightsCopy(weights,weightsSwap,neurons,nlayers,ninputs);

		/* Update layer */
		updateLayer(weights,alpha,weightsPast, lrate, gs, \
		bias[nlayers-1], yout, nlayers-1, neurons[nlayers-1], \
		neurons[nlayers-2]);

		/* Update layers */
		for(i=nlayers-2; i>=0; i--)
		{
			/* SUM GsW */
			sumWtGs(wgs,weights,gs,i+1,neurons[i+1],neurons[i]);	
			
			/* Derivative of the activation function = df */
			dActivation(df,yout,i,neurons[i],actv);			
	
			/* Local gradient. */
			/* df * sum of (next layer G * next layer weights) */
			gradient(gs,wgs,df,i,neurons[i]);		

			/* Update layer. */
			if(i)
			{
				updateLayer(weights,alpha,weightsPast, \
				lrate, gs, bias[i], yout, i, neurons[i], \
				neurons[i-1]);
			}
			else
			{
				updateLayer0(weights,alpha,weightsPast, \
				lrate, gs, bias[i], x, xidx[ex-1], \
				neurons[i], ninputs);
			}
		}

		/* Save past weights */
		weightsCopy(weightsSwap,weightsPast,neurons,nlayers,ninputs);

		/* Save output of the examples */
		for(i=0; i<neurons[nlayers-1]; i++)
			youtLastLayers[xidx[ex-1]][i] = yout[nlayers-1][i];

		/* Mean Square Error. */
		if (ex == examples)
		{
			/* MSE */
			mse = mseb(ref,youtLastLayers, \
			examples,neurons[nlayers-1]);
			/* Change order of training set */
			vperm(xidx,examples);
			/* Save history of MSE */
			mse_hist[mse_counter] = mse;
			mse_counter += 1;
		}

		/* Display the progress */
#ifdef DEBUG_MODE
		if( (progress % displayStep) == 0 )
		{
			printf("%.2f%% of maximum iteration.\n", 
			(float) (counter*100)/maxIteration);
			printf("MSE: %.4e\n",mse);
		}
		progress += 1;
#endif
	}

	/* Add in the position '0' the 'mse_counter'-1 
	 * to identify the last position of history
	 */
	mse_hist[0] = mse_counter-1;

	/* Deallocate memory */
	// weightsPast
	// yout
	// youtLastLayers
	// error
	// gs
	// df
	// wgs
	// xidx

	/* Return history of MSE */
	return mse_hist;
}

/* Output of MLP */
void outMLP(double *** weights, training * trainingData, \
char * activation, double ** in, int pos, double * out)
{
	int nlayers = trainingData->nlayers;
	int * neurons = trainingData->neurons;
	int ninputs = trainingData->ninputs;
	double alpha = trainingData->alpha;
	double * bias = trainingData->bias;

	int layer;
	int i;
	int actv = getActv(activation);

	/* Memory for layers outputs */
	double ** yout = (double**) malloc(sizeof(double*)*nlayers);
	for(i=0; i<nlayers; i++)
		yout[i] = (double*) malloc(sizeof(double)*neurons[i]);

	/* Propagation */	
		
	/* Out of first layer */
	layerOut0(yout,bias[0],in,pos,ninputs,\
	weights,neurons[0],actv);

	for(layer=1; layer<nlayers; layer++)
	{
		/* Out of "layer" layer */
		layersOut(yout,bias[layer],neurons[layer-1], \
		weights,layer,neurons[layer],actv);
	}

	/* Copy output of last layer to out */
	for(i=0; i<neurons[nlayers-1]; i++)
		out[i] = yout[nlayers-1][i];

	/* Deallocate 'yout' memory */
	for(i=0; i<nlayers; i++)
		free(yout[i]);

	free(yout);
}

/* Save training data and weights of MLP in a file */
void saveMLP(char * filename, training * trainingData, \
double *** weights)
{

}

/* Load the examples, references and configuration from files. */
int loadExamplesFromFile(char * inputs, char * outputs, \
char * conf, training * trainingData);

/* Print the neural network */
void printMLP(double *** weights, training * trainingData)
{
	int nlayers = trainingData->nlayers;
	int * neurons = trainingData->neurons;
	int ninputs = trainingData->ninputs;
	double * bias = trainingData->bias;

	int l;
	int n;
	int w;
	printf("\n--------------------\n");
	printf("Neural Network\n\nInputs: %d\nLayers: %d",\
	ninputs,nlayers);
	printf("\nOutputs: %d\n\n", neurons[nlayers-1]);	
	
	for(l=0; l<nlayers; l++)
	{
		printf("Layer %d\nBias: %.6f\n",l,bias[l]);
		for(n=0; n<neurons[l]; n++)
		{
			printf(" Neuron %d\n",n);
			if(l)
			{
				for(w=0; w<(neurons[l-1]+1); w++)
					printf("  Weight %d: %.6f\n",w,weights[l][n][w]);
			}
			else
			{
				for(w=0; w<(ninputs+1); w++)
					printf("  Weight %d: %.6f\n",w,weights[l][n][w]);
			}
		}
	}
	printf("\n--------------------\n");
}

