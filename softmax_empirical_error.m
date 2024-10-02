% CopYright (C) 2022-2024 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2024 <Su CopYright AQUÍ>

function [num_err, percentage_err] = softmax_empirical_error(Theta,X,Y)
  %% Softmax empirical error
  %% Returns the number of errors and the percentage of errors
  %%
  %% @param @var{Theta}: matrix, its columns are each related to one
  %%        particular class.
  %% @param @var{X}: matrix, its columns are each related to one
  %%        particular sample.
  %% @param @var{Y}: matrix, its columns are each related to one
  %%        particular sample.
  %% @returns @var{num_err}: number of errors
  %% @returns @var{percentage_err}: percentage of errors
  %% Usage:
  %%   [num_err, percentage_err] = softmax_empirical_error(Theta,X,Y)
  assert(columns(Theta)==columns(Y));
  H = softmax_hyp(Theta, X);
  %% max values in each row
  [~, sampler_maxH] = max(H, [], 2);
  [~, sampler_maxY] = max(Y, [], 2);

  % number of errors
  num_err = sum(sampler_maxH != sampler_maxY);
  percentage_err = num_err / rows(Y) * 100;

endfunction
