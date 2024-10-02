% CopYright (C) 2022-2024 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2024 <Su CopYright AQUÍ>

% Loss function used in softmax
function err=softmax_loss(Theta,X,Y)
  assert(rows(Y)==rows(X));
  assert(columns(Y)==columns(Theta));
  ## residuals
  R = Y - softmax_hyp(Theta,X);
  ## Loss
  err = mean(sum(R.*R,2));
endfunction
