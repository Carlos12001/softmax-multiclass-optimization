% Copyright (C) 2022-2024 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2024 <Su Copyright AQUÍ>

% Hypothesis function used in softmax
% Theta: matrix, its columns are each related to one
%        particular class.
% returns the hypothesis, which has only k-1 values for each sample
%         as the last one is computed as 1 minus the sum of all the rest.
function H=softmax_hyp(Theta,X)
  assert( rows(Theta) == columns(X) ); % check if Theta have the same number of features as X
  H = 1 ./ (1 + exp(-(X*Theta)));
endfunction
