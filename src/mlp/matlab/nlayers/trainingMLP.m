%% n-Layer Multilayer Perceptron Training.
% [W,E] = trainingMLP(neurons,a,bias,x,yref,lr,error,maxIt)

% % Copyright (c) 2015, Augusto Damasceno.
% % All rights reserved.
% % Redistribution and use in source and binary forms, with or without modification,
% % are permitted provided that the following conditions are met:
% %   1. Redistributions of source code must retain the above copyright notice,
% %      this list of conditions and the following disclaimer.
% %   2. Redistributions in binary form must reproduce the above copyright notice,
% %      this list of conditions and the following disclaimer in the documentation
% %      and/or other materials provided with the distribution.
% % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% % ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% % WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
% % IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
% % INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
% % BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
% % OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
% % WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% % ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
% % OF SUCH DAMAGE.

function [W,E] = trainingMLP(neurons,a,bias,x,yref,lr,error,maxIt)
%% n-Layer Multilayer Perceptron Training.

% neurons = 1 X layers.
% a = momentum constant.
% bias(layer) = 1 X layers.
% x = examples X inputs.
% yref = examples X desired outputs.
% lr = learning-rate.
% error = acceptable error.
% maxIt = maximum iteration.

%% Configuration and architecture.

[~,layers] = size(neurons);
[examples,nInputs] = size(x);
[~,nOutputs] = size(yref);

% Memory of weights: layers X neurons X weights.
wm = rand(layers,max(neurons),max(neurons)+1);

% Memory of past weights.
wmPast = zeros(layers,max(neurons),max(neurons)+1);

% Memory of outs.
ym = rand(layers,max(neurons));

%% Training. 

ex = 0;
counter = 0;
progress = 0;
displayStep = ceil(0.05*maxIt);

% Memory of MLP Outputs.
ys = zeros(examples,nOutputs);

% Mean Square Error.
mse = +Inf;

% Memory of Mean Square Error.
mse_hist = zeros(ceil(maxIt/examples),1);
mse_counter = 1;

% Random order of inputs - desired outputs.
xidx = randperm(examples);

while (mse > error && counter < maxIt)

    % Iteration Number.
    counter = counter + 1;
    
    % Number of training example.
    ex = mod(ex,examples) + 1;
    
    % Propagation.
    
    % Copy weights of first layer from memory of weights.
    w1(1:neurons(1),1:(nInputs+1)) = wm(1,1:neurons(1),1:(nInputs+1));
    % Out of first layer.
    y = sigmoid( ( [bias(1) x(xidx(ex),:)]*w1' ) ,1) ;
    % Save out of first layer to memory of outs.
    ym(1,1:neurons(1)) = y;
    for k=2:layers
        clear wk;
        % Copy weights of k layer from memory of weights.
        wk(1:neurons(k),1:(neurons(k-1)+1)) = wm(k,1:neurons(k),1:(neurons(k-1)+1));
        % Out of k layer.
        y = sigmoid( [bias(k) y]*wk' ,1);
        % Save out of k layer to memory of outs.
        ym(k,1:neurons(k)) = y;
    end
    
    % Backpropagation.
    
    % Neuron Update = learning-rate*G*y(layer-1)
    
    % Copy out of last layer from memory of outs.
    % y = ym(layers,1:neurons(layers));
    % Error = yref - y
    E = yref(xidx(ex),:) - y;
    % Derivative of the activation function = df
    df = y.*(1-y);
    % Local gradient. Last layer = Error * df.
    Gs = E .* df;
    % Copy out of previous layer from memory of outs.
    ypre = ym(layers-1,1:neurons(layers-1));
    clear w;
    clear wPast;
    % Copy weights of last layer from memory of weights.
    w(1:neurons(layers),1:(neurons(layers-1)+1)) = wm(layers,1:neurons(layers),1:(neurons(layers-1)+1));
    % Copy past weights of last layer from memory of past weights.
    wPast(1:neurons(layers),1:(neurons(layers-1)+1)) = wmPast(layers,1:neurons(layers),1:(neurons(layers-1)+1));
    % Save weights before update.
    wtmp = w;
    % Update last layer.
    w = w + a*wPast + lr*Gs'*[bias(layers) ypre];
    % Save new weights of last layer.
    wm(layers,1:neurons(layers),1:(neurons(layers-1)+1)) = w;
    % Save weights before update to memory of past weights.
    wmPast(layers,1:neurons(layers),1:(neurons(layers-1)+1)) = wtmp;
    
    for k=(layers-1):-1:2
        clear wnext;
        % Copy weights of k next layer from memory of weights.
        wnext(1:neurons(k+1),1:(neurons(k)+1)) = wm(k+1,1:neurons(k+1),1:(neurons(k)+1));
        % Copy out of k layer from memory of outs.
        y = ym(k,1:neurons(k));
        % SUM GsWnext
        WtGs = wnext'*Gs';
        % Derivative of the activation function
        df = (y.*(1-y));
        % Local gradient. Hidden layers: df * sum of (next layer G * next layer weights)
        Gs = df.*WtGs(2:end,:)';
        % Copy out of previous layer from memory of outs.
        ypre = ym(k-1,1:neurons(k-1));
        clear w;
        clear wPast;
        % Copy weights of k layer from memory of weights.
        w(1:neurons(k),1:(neurons(k-1)+1)) = wm(k,1:neurons(k),1:(neurons(k-1)+1));
        % Copy past weights of k layer from memory of past weights.
        wPast(1:neurons(k),1:(neurons(k-1)+1)) = wmPast(k,1:neurons(k),1:(neurons(k-1)+1));
        % Save weights before update.
        wtmp = w;
        % Update k layer.
        w = w + a*wPast + lr*Gs'*[bias(k) ypre];
        % Save new weights of k layer.
        wm(k,1:neurons(k),1:(neurons(k-1)+1)) = w;
        % Save weights before update to memory of past weights.
        wmPast(k,1:neurons(k),1:(neurons(k-1)+1)) = wtmp;
    end
    
    clear wnext;
    % Copy weights of second layer from memory of weights.
    wnext(1:neurons(2),1:(neurons(1)+1)) = wm(2,1:neurons(2),1:(neurons(1)+1));
    % Copy out of first layer from memory of outs.
    y = ym(1,1:neurons(1));
    % SUM GsWnext
    WtGs = wnext'*Gs';
    % Derivative of the activation function
    df = (y.*(1-y));
    % Local gradient. Hidden layers: df * sum of (next layer G * next layer weights)
    Gs = df.*WtGs(2:end,:)';
    clear w;
    clear wPast;
    % Copy weights of first layer from memory of weights.
    w(1:neurons(1),1:nInputs+1) = wm(k,1:neurons(1),1:nInputs+1);
    % Copy past weights of first layer from memory of past weights.
    wPast(1:neurons(1),1:nInputs+1) = wmPast(1,1:neurons(1),1:nInputs+1);
    % Save weights before update.
    wtmp = w;
    % Update first layer.
    w = w + a*wPast + lr*Gs'*[bias(k) x(xidx(ex),:)];
    % Save new weights of first layer.
    wm(1,1:neurons(1),1:nInputs+1) = w;
    % Save weights before update to memory of past weights.
    wmPast(1,1:neurons(1),1:nInputs+1) = wtmp;
    
    % Mean Square Error.
    ys(xidx(ex),:) = ym(layers,1:neurons(layers));
    if ex == examples
        mse = mseb(ys,yref);
        % Change order of inputs - desired outputs.
        xidx = randperm(examples);
        % Save history of MSE.
        mse_hist(mse_counter) = mse;
        mse_counter = mse_counter + 1;
    end
    
    % Display the progress.
    if mod(progress,displayStep) == 0
    	message = sprintf('%.2f%% of maximum iteration.\n',(counter*100)/maxIt);
    	disp(message);
        message = sprintf('MSE: %.4e\n',mse);
        disp(message);
    	drawnow();
    end
    progress = progress+1;
end

fprintf('\nTotal Iterations: %d\nError: %.4e\n',counter,mse);    

assignin('base','mse_hist',mse_hist);

W = wm;
E = mse;
end
