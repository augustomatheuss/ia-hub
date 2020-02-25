function animation(w1_hist,w2_hist,neurons)
%% Animation from history of weights.

figure(3);
[s,~] = size(w1_hist);

i=1;
while i<s
  disp(i)
  clf(3); 
  neighborhood(w1_hist(i,:),w2_hist(i,:),sqrt(neurons),sqrt(neurons));
  drawnow;
  if i < (s-500*neurons)/4
    i = i+ceil(0.001*s);
  else
    i = i+ceil(0.05*s);
  end
end

clf(3);
neighborhood(w1_hist(s,:),w2_hist(s,:),sqrt(neurons),sqrt(neurons));
drawnow;

end
