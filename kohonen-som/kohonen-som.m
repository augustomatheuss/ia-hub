%% Kohonen Self-Organizing Map.
% [W,X,Y,D] = kohonen(neurons,inputs,nodes,source,error)

% % Copyright (c) 2014,2015 Augusto Damasceno.
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

function [W,X,Y,D] = kohonen(neurons,inputs,nodes,source,error)
%% Kohonen Self-Organizing Map.

% Neuron = w(:,j) .: Inputs X Neurons
% Source = SourceNodes X Inputs
% Neuron Coordinate =  wx(1,j) , wy(1,j)

% Neighborhood Function = hj,i(x)
% Size of Neighborhood = S
% Learning-rate = N
% Synaptic-weight Vectors = w
% Input = source
% Winning Neuron = i(x)

% i(x) = arg min j || x - wj || j = 1,2,...,l
% .: w'x Maximum inner product or minor euclidean distance.
% hj,i(x) = exp( -1 * ( distj,i^2 ) / ( 2*(S^2) ) ) n = 0,1,2,...
% distj,i^2 = || rj - ri || ^ 2
% S = S0 * exp( - n/t1 )  n = 0,1,2,...
% S0 = constant
% t1 = time constant
% N = N0*exp( - n/t2 ) n = 0,1,2,...
% N0 = constant
% t2 = time constant
% wj(n+1) = w(n) + N*hj,i(x)*( x-wj(n) )


% Inicialization
    % Parameters.
    Neurons = neurons;
    Inputs = inputs;
    SourceNodes = nodes;
    IterationsOrdering = 8000;
    MaximumConvergence = 50000*Neurons;
    Dim = sqrt(Neurons);
    S0 = Dim;
    T1 = IterationsOrdering/log(S0);
    N0 = 0.1;
    N_limit = 0.01;
    S_limit = 1;
    T2 = IterationsOrdering/log(N0/N_limit);
    
    % Weight Vectors. 
    w = rand(Inputs,Neurons);
    
    % Weight Coordinates.
    [rows,columns] = meshgrid(1:Dim);
    wx = zeros(1,Neurons);
    wy = zeros(1,Neurons);
    wx(1,1:Neurons) = rows(1:Neurons);    
    wy(1,1:Neurons) = columns(1:Neurons);
    
    % Input Space.
    x = source;
    
    disp('Start of Ordering Phase.');
    disp(char(10));
    drawnow();

% Ordering Phase.

    ex = 0;
    dist = zeros(1,Neurons);
    dif = zeros(Inputs,Neurons);
    h = zeros(1,Neurons);
    for n=0:(IterationsOrdering-1)
        % Sampling.
        
        ex = mod(ex,SourceNodes) + 1;
        
        % Competitive Process: Similarity Matching.
        
        % Repeat Weight Vector of Input to Match With w Matrix Dimensions.
        xn_rep = repmat(x(ex,:)',1,Neurons);
        
        % Search minor euclidean distance.
        dist(1,1:Neurons) = ( sum( (w - xn_rep ).^2 ) ).^0.5;
        [~,winning] = min(dist);

        % Cooperative Process and Updating.
        
        % Update the size of neighborhood.
        S = S0*exp(-1 * ( n / T1 ) );
        % Update the learning-rate.
        N = N0*exp(-1 * ( n / T2 ) );
        % Inferior limit of learning-rate.
        if N < N_limit
            N = N_limit;
        end
        % Inferior limit of size of neighborhood.
        if S < S_limit
            S = S_limit;
        end
        % Euclidean distance with winning neuron to square.
        distp2 = (( wx - wx(1,winning) ).^2) + (( wy - wy(1,winning) ).^2);
        % Difference with input.
        dif(:,1:Neurons) =  xn_rep(:,1:Neurons) - w(:,1:Neurons);
        % Neighborhood Function.
        h = exp(-1 * ( distp2 ./ (2*S*S) ) );
        % Update weights.
        h_rep = repmat(h,Inputs,1);
        w(:,1:Neurons) = w(:,1:Neurons) + N.*h_rep.*dif(:,1:Neurons);
    end

    disp('End of Ordering Phase.');
    disp(char(10));
    disp('Start of Convergence Phase.');
    disp(char(10));
    drawnow();  
        
% Convergence Phase.

    % Maintain learning-rate in a positive small value.
    N = N_limit;
    % Maintain neighborhood containing nearest neighbors only.
    S = S_limit;
    
    ex = 0;
    verify = true;
    dw = +Inf;
    progress = 0;
    displayStep = ceil(0.05*MaximumConvergence);
    error_flag = 0;
    while verify
        n = n+1;
        history = w;
        
        % Sampling.
        
        ex = mod(ex,SourceNodes) + 1;
        
        % Competitive Process: Similarity Matching.
        
        % Repeat Weight Vector of Input to Match With w Matrix Dimensions.
        xn_rep = repmat(x(ex,:)',1,Neurons);
        
        % Search minor euclidean distance.
        dist(1,1:Neurons) = ( sum( (w - xn_rep ).^2 ) ).^0.5;
        [~,winning] = min(dist);

        % Cooperative Process and Updating.

        % Euclidean distance with winning neuron to square.
        distp2 = (( wx - wx(1,winning) ).^2) + (( wy - wy(1,winning) ).^2);
        % Difference with input.
        for j=1:Neurons
            dif(:,j) = x(ex,:)' - w(:,j);
        end
        % Neighborhood Function.
        h = exp(-1 * ( distp2 ./ (2*S*S) ) );
        % Update weights.
        for j=1:Neurons
             w(:,j) = w(:,j) + N.*h(1,j).*dif(:,j);
        end
        
        % If there is no noticeable changes in feature map.
       
        if mod(n+1,SourceNodes) == 0;
            dw_hist = dw;
            dw = norm(history-w);
            if dw_hist < dw
                message = sprintf('Changes rising. n = %d\nLast Change = %.8d\nCurrent Change = %.8d\n',n,dw_hist,dw);
                disp(message);
                drawnow();
                verify = false;
                progress = 500;
            end
            if dw_hist == dw
                error_flag = error_flag + 1;
            end
            if (dw < error) || (n>(MaximumConvergence)) || (error_flag > 10)
                verify = false;
            end
            
            % Display the progress.
            if mod(progress,displayStep)  == 0
                message = sprintf('%.2f%% of maximum iteration of Convergence Phase.\n',(n*100)/MaximumConvergence);
                disp(message);
                drawnow();
            end
            progress = progress+1;
        end
        
    end
    
    message = sprintf('End of Convergence Phase. Iteration: %d',n);
    disp(message);
    disp(char(10));
    message = sprintf('Change: %.8d\n', abs(norm(history-w)) );
    disp(message);
    drawnow();
    
   W = w;
   X = wx;
   Y = wy;
   D = dw;
end
