%% n-Layer Multilayer Perceptron Output.
% [O] = outMLP(neurons,bias,input,w)

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

function [O] = outMLP(neurons,bias,input,w)
%% n-Layer Multilayer Perceptron Output.

% bias(layer) = 1 X layers.
% input = 1 X inputs.

%% Configuration and architecture.

[~,layers] = size(neurons);
[~,nInputs] = size(input);

% Propagation.
    
% Copy weights of first layer from memory of weights.
w1(1:neurons(1),1:(nInputs+1)) = w(1,1:neurons(1),1:(nInputs+1));
% Out of first layer.
y = sigmoid( ( [bias(1) input]*w1' ) ,1) ;
for k=2:layers
    clear wk;
	% Copy weights of k layer from memory of weights.
	wk(1:neurons(k),1:(neurons(k-1)+1)) = w(k,1:neurons(k),1:(neurons(k-1)+1));
	% Out of k layer.
	y = sigmoid( [bias(k) y]*wk' ,1);
end
    
O = y;

end
