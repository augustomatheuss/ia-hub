%% Simulation and analysis of Kohonen Self-Organizing Map in two dimensions.

% % Copyright (c) 2014, 2015 Augusto Damasceno.
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

% Neighborhood Function = hj,i(x)
% Size of Neighborhood = S
% Learning-rate = N
% Synaptic-weight Vectors = w1,w2 
% Input Vectors = x1,x2
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

%% Inicialization
    
    % Parameters.
    Neurons = 100;
    SourceNodes = 2000;
    IterationsOrdering = 4000;
    Dim = sqrt(Neurons);
    S0 = Dim;
    T1 = IterationsOrdering/log(S0);
    N0 = 0.1;
    N_limit = 0.01;
    S_limit = 1;
    T2 = IterationsOrdering/log(N0/N_limit);
    
    % Weight Vectors.
    w1 = rand(1,Neurons);
    w2 = rand(1,Neurons);
    
    % Neurons' Coordinates.
    [rows,columns] = meshgrid(1:Dim);
    wx = zeros(1,Neurons);
    wy = zeros(1,Neurons);
    wx(1,1:Neurons) = rows(1:Neurons);    
    wy(1,1:Neurons) = columns(1:Neurons);
    
    % Input Space.
    x1 = rand(SourceNodes,1); 
    x2 = rand(SourceNodes,1);
    
    figure(1);
    
    % Plot graph with initial weights.
    subplot(2,2,2); 
    neighborhood(w1,w2,Dim,Dim);
    xlabel('w1');
	ylabel('w2');
    title('Weights in Initial State','FontSize',14);
    axis([0,1,0,1]);
    
    % Plot graph with input space.
    subplot(2,2,1); 
    plot(x1(:,1),x2(:,1),'*');
    xlabel('x1');
	ylabel('x2');
    title('Input Space','FontSize',14);
    axis([0,1,0,1]);
    
    % Summary of parameters.
    subplot(2,2,3);
    plot(0,0,'w.');
    s = sprintf('S0 = %.2f\n\n',S0);
    s = sprintf('%sT1 = %.2f\n\n',s,T1);
    s = sprintf('%sN0 = %.2f\n\n',s,N0);
    s = sprintf('%sT2 = %.2f\n\n',s,T2);
    s = sprintf('%sNeurons = %d\n\n',s,Neurons);
    s = sprintf('%sSource Nodes = %d\n\n',s,SourceNodes);
    s = sprintf('%sIterations of Ordering Phase = %d',s,IterationsOrdering);
    text(0.1,0.5,s);
    axis([0,1,0,1]);
    title('Parameters','FontSize',14);
    drawnow;
    
    figure(2);
    
    % Plot graph with weights after ordering phase.
    subplot(2,2,1)
    plot(0,0,'w.');
    leg1 = legend('Wait the end of ordering phase',0);
    xlabel('w1');
    ylabel('w2');
    title('End of Ordering Phase','FontSize',14);
    drawnow;
    
    % Plot graph with weights after convergence phase.
    subplot(2,2,3)
    plot(0,0,'w.');
    leg2 = legend('Wait the end of convergence phase',0);
    xlabel('w1');
    ylabel('w2');
    title('End of Convergence Phase','FontSize',14);
    drawnow;
    
    % History of size of neighborhood and learning-rate.
    S_hist = zeros(1,IterationsOrdering+(Neurons*500));
    N_hist = zeros(1,IterationsOrdering+(Neurons*500));
    
    % History of weight vectors.
    w1_hist = zeros(IterationsOrdering+(Neurons*500),Neurons);
    w2_hist = zeros(IterationsOrdering+(Neurons*500),Neurons);
    w1_hist(1,:) = w1;
    w2_hist(1,:) = w2;
    
    % History of winning neurons.
    win_hist = zeros(1,IterationsOrdering+(Neurons*500));
    
%% Ordering Phase.

    ex = 0;
    for n=0:(IterationsOrdering-1)
        % Sampling.
        
        ex = mod(ex,SourceNodes) + 1;
        
        % Competitive Process: Similarity Matching.
        
        % Search minor euclidean distance.
        dist = sqrt(((w1-x1(ex,1)).^2) + ((w2-x2(ex,1)).^2));
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
        dif1 = x1(ex,1) - w1; 
        dif2 = x2(ex,1) - w2;
        % Neighborhood Function.
        h = exp(-1 * ( distp2 ./ (2*S*S) ) );
        % Update weights.
        w1 = w1 + N*h.*dif1;
        w2 = w2 + N*h.*dif2;  
        % Save history of size of neighborhood and learning-rate.
        S_hist(1,n+1) = S;
        N_hist(1,n+1) = N;
        % Save history of weight vectors.
        w1_hist(n+2,:) = w1;
        w2_hist(n+2,:) = w2;
        % Save history of winning neurons.
        win_hist(1,n+1) = winning;
    end
        
    % Plot graph with weights in the end of ordering phase.
    subplot(2,2,1)
    delete(leg1);
    neighborhood(w1,w2,Dim,Dim);
    
%% Convergence Phase.

    % Maintain learning-rate in a positive small value.
    N = N_limit;
    % Maintain neighborhood containing nearest neighbors only.
    S = S_limit;
    
    ex = 0;
    for n=(IterationsOrdering):((Neurons*500)-1+IterationsOrdering)
        % Sampling.
        
        ex = mod(ex,SourceNodes) + 1;
        
        % Competitive Process: Similarity Matching.
        
        % Search minor euclidean distance.
        dist = sqrt(((w1-x1(ex,1)).^2) + ((w2-x2(ex,1)).^2));
        [~,winning] = min(dist);
        
        % Cooperative Process and Updating.
        
        % Euclidean distance with winning neuron, to square.
        distp2 = (( wx - wx(1,winning) ).^2) + (( wy - wy(1,winning) ).^2);
        % Difference with input.
        dif1 = x1(ex,1) - w1; 
        dif2 = x2(ex,1) - w2;
        % Neighborhood Function.
        h = exp(-1 * ( distp2 ./ (2*S*S) ) );
        % Update weights.
        w1 = w1 + N*h.*dif1;
        w2 = w2 + N*h.*dif2;  
        % Save history of size of neighborhood and learning-rate.
        S_hist(1,n+1) = S;
        N_hist(1,n+1) = N;
        % Save history of weight vectors.
        w1_hist(n+2,:) = w1;
        w2_hist(n+2,:) = w2;
        % Save history of winning neurons.
        win_hist(1,n+1) = winning;
    end

    % Plot graph with weights in the end of convergence phase.
    subplot(2,2,3)
    delete(leg2);
    neighborhood(w1,w2,Dim,Dim);
    
    % Plot graph with history of learning-rate.
    subplot(2,2,2);
    plot(0:1:IterationsOrdering+(Neurons*500)-1,N_hist);
    ylabel('N');
    xlabel('n');
    grid;
    title('Learning-rate','FontSize',14);
    axis([0,IterationsOrdering+(Neurons*500)-1,0,N0]);
    
    % Plot graph with history of size of neighborhood.
    subplot(2,2,4);
    plot(0:1:IterationsOrdering+(Neurons*500)-1,S_hist);
    ylabel('S');
    xlabel('n');
    grid;
    title('Size of Neighborhood','FontSize',14);
    axis([0,IterationsOrdering+(Neurons*500)-1,0,S0]);
    drawnow;
