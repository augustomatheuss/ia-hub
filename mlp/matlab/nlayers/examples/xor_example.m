%% Learning XOR logical operation

% Configuration
neurons = [3 4 1];
a = 1e-4;
bias = [-1 -1 -1];
x = [0 0;0 1;1 0; 1 1];
yref = [0 ;1 ;1 ;0 ];
lr = 0.375;
error = 1e-20;
maxIt = 1e5;

% Processing
[W,E] = trainingMLP(neurons,a,bias,x,yref,lr,error,maxIt);

% Display Infos
disp('XOR')
disp('Input [0 0]');
outMLP(neurons,bias,[0 0],W)
disp('Input [0 1]');
outMLP(neurons,bias,[0 1],W)
disp('Input [1 0]');
outMLP(neurons,bias,[1 0],W)
disp('Input [1 1]');
outMLP(neurons,bias,[1 1],W)

% Plot MSE
semilogx(mse_hist)
ylabel('MSE');
xlabel('Iteration');
title('Xor Example','FontSize',14);
