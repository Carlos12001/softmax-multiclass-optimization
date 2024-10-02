% Copyright (C) 2022-2024 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 4
% (C) 2024 <Su Copyright AQUÍ>

% Set default values for all plots
clc; clear all; close all;
set(0, "DefaultAxesFontSize", 32); % Make size labels bigger
set(0, "DefaultAxesTitleFontSizeMultiplier", 1.25);
labels_x = {"Bias", "Culmen Length (mm)", "Culmen Depth (mm)", ...
            "Flipper Length (mm)", "Body Mass (g)"};
labels_y = {"Adelie", "Chinstrap", "Gentoo"};
fig_count = 1;
part_4_5 = true;
part_6_7_8_9_10 = true;

% Softmax testbench

[Xtr,Ytr,Xte,Yte,names] = loadpenguindata("species");

Xtr = [ ones(rows(Xtr),1) Xtr];
Xte = [ ones(rows(Xte),1) Xte];


assert(columns(Xtr)==length(labels_x));
assert(columns(Ytr)==length(labels_y));

Theta0=rand(columns(Xtr),columns(Ytr))-0.5; ## Random starting point



str_normalizer =  "normal";
%% Normalize input values
nx = normalizer(str_normalizer);
Xtr_normal = nx.fit_transform(Xtr);
Xte_normal = nx.transform(Xte);


## Initial configuration for the optimizer
## Use 10% of the data as minibatch
opt=optimizer("method","sgd",
              "mbmode","norep",
              "minibatch",floor(rows(Xtr_normal)*0.1),
              "maxiter",599,
              "alpha",0.05,
              "beta1",0.9,
              "beta2",0.999,
              "show","progress");

if part_4_5
  ### PART FOUR and PART FIVE
  printf("\n\n##### PART FOUR and PART FIVE #####\n\n");

  ## Train the model
  figure(fig_count,"name","Loss Evolution");
  hold on;

  ## Methods Optimization
  methods={"sgd","momentum","rmsprop","adam","batch"};
  % methods = methods(4);
  Theta = zeros(rows(Theta0),columns(Theta0),length(methods));
  ## Train the model
  for i=1:length(methods)
    method=methods{i};
    printf("Training with method '%s'.\n",method);
    msg = sprintf(";%s;",method); ## use method in legends
    try
      opt.configure("method",method); ## Just change the method
      [ts, errs] = minimize(opt,@softmax_loss,Theta0,Xtr_normal,Ytr);
      Theta(:,:,i) = ts{end};
      display(Theta(:,:,i));

      [num_errors, percentage_error]=softmax_empirical_error(Theta(:,:,i),
                                                                Xtr_normal,Ytr);
      printf("Training error: %d / %d (%d %%)\n",  num_errors, rows(Ytr),
                                                  percentage_error);
      
      [num_errors, percentage_error]=softmax_empirical_error(Theta(:,:,i),
                                                                Xte_normal,Yte);
      printf("Test error: %d / %d (%d %%)\n",  num_errors, rows(Yte), 
                                            percentage_error);
      
      figure(fig_count);
      plot(errs,msg,"linewidth",4);
    catch
      printf("\n### An error ocurred in '%s': ###\n %s\n\n",
            method,lasterror.message);
    end_try_catch
  endfor
  figure(fig_count++);
  xlabel("Iteration");
  ylabel("Loss");
  grid on;
  hold off;
endif


if part_6_7_8_9_10
  printf("\n\n##### PART SEVEN - EIGHT - NINE - TEN #####\n\n");
  opt.configure("method","adam",
                "maxiter",99,
                "mbmode","norep",
                "minibatch",floor(rows(Xtr_normal)*0.1),
                "alpha",0.05,
                "beta1",0.9,
                "beta2",0.999,
                "show","progress"
              );

  %% Find the most important features - Part 6
                
  aumount_features = 2;
  with_bias = true; ## Add bias to the features to find the most important features

  ## Get all possible combinations
  init_combination = 1;
  if !with_bias init_combination = 2; endif
  
  comb = nchoosek(init_combination:columns(Xtr_normal), aumount_features);

  ## Initial configuration for the optimizer
  opt.configure("method","adam");

  ## Test all possible combinations
  list_percentage_error = [];
  Theta = zeros(aumount_features,columns(Theta0),length(comb));

  for i=1:rows(comb)
    %% Get the current combination of features
    Xtr_temp = Xtr_normal(:,comb(i,:));
    Xte_temp = Xte_normal(:,comb(i,:));
    Theta0_temp = Theta0(comb(i,:),:);

    printf("Combination: ");
    display(comb(i,:));

    try
      [ts,errs]=opt.minimize(@softmax_loss,Theta0_temp,Xtr_temp,Ytr);
      Theta(:,:,i) = ts{end};

      [num_errors, percentage_error]=softmax_empirical_error(Theta(:,:,i),
      Xte_temp,Yte);

      list_percentage_error = [list_percentage_error percentage_error];
      
      printf("Test error: %d / %d (%d %%)\n",
        num_errors,rows(Yte),percentage_error);
    
    catch
      printf("\n### Error testing combination: \t");
      display(comb(i,:));
      printf("\t: ###\n %s\n\n", lasterror.message);
    end_try_catch
  endfor


  %% Find the features with least error
  min_percentage_error = min(list_percentage_error);
  index_min = find(list_percentage_error==min_percentage_error);
  index_min = index_min(:);
  assert(columns(index_min) == 1);

  printf("\n\nBest combination are: \n\n");
  for i = 1:rows(index_min)
    printf("Combination: \n");
    display(comb(index_min(i),:));
    printf("[%s, %s]\n",  labels_x{comb(index_min(i),1)}, 
                          labels_x{comb(index_min(i),2)}
          );
    printf("percentage error: %d %% \n###########\n\n", 
    list_percentage_error(index_min(i)));
  endfor

  %% Plot the most important features - Part 7

  %% Use the first combination with least error
  Xtr_temp = Xtr(:,comb(index_min(1),:));
  Xte_temp = Xtr(:,comb(index_min(1),:));
  best_Theta = Theta(:,:,index_min(1));

  %% Plot the features
  feature1_range = linspace(min(Xte_temp(:,1)), max(Xte_temp(:,1)), 100);
  feature2_range = linspace(min(Xte_temp(:,2)), max(Xte_temp(:,2)), 100);
  [feature1_grid, feature2_grid] = meshgrid(feature1_range, feature2_range);

  %% Create the design matrix for the grid
  X = [feature1_grid(:), feature2_grid(:)];

  %% Normalize the features
  %% Note: I can use the same normalizer nx but is the same cut the 
  %% features an then use a new normalizer and transform them. 
  %% It will generate the same results
  nx_temp = normalizer(str_normalizer);
  Xtr_temp_normal = nx_temp.fit_transform(Xtr_temp);
  Xte_temp_normal = nx_temp.transform(Xte_temp);
  X_normal = nx_temp.transform(X);

  %% Calculate the probabilities
  h = softmax_hyp(best_Theta, X_normal);

  %% Reshape the probabilities to match the grid
  probabilities = zeros(rows(feature1_grid),columns(feature1_grid),columns(h));
  %% Plot the surfaces
  for i = 1:columns(h)
    probabilities(:,:,i) = reshape(h(:,i), size(feature1_grid));

    figure(fig_count++, "name", sprintf("Probability of Penguin being %s",
      labels_y{i}));
    surf(feature1_grid, feature2_grid, probabilities(:,:,i));
    xlabel(sprintf("%s", labels_x{comb(index_min(1),1)}));
    ylabel(sprintf("%s", labels_x{comb(index_min(1),2)}));
    zlabel(sprintf("p(y=%s|x)", labels_y{i}));
    title(sprintf("Probability of Penguin being %s", labels_y{i}));
    colorbar;
  endfor

  %% Plot the Winner Classes -  Part 8
  %% Assuming h is your matrix of class probabilities (points x classes)
  [~, class_indices] = max(h, [], 2);  % Find the class with the highest probability for each point

  %% Reshape to match the grid
  class_matrix = reshape(class_indices, size(feature1_grid));

  %% Create a color palette
  % This is just an example, adjust the number of colors based on your classes
  color_map = [1,0,0; 0,1,0; 0,0,1];

  %% Convert indices to RGB image
  rgb_image = ind2rgb(class_matrix, color_map);

  %% Visualize the class regions
  figure(fig_count++, "name", "Winning Class Regions");
  image(feature1_range, feature2_range, rgb_image); % Adjust axes if necessary
  set(gca, "YDir", "normal"); % Ensure the Y-axis direction is correct
  xlabel(sprintf("%s", labels_x{comb(index_min(1),1)}));
  ylabel(sprintf("%s", labels_x{comb(index_min(1),2)}));
  title("Winning Class Regions");



  % Adda legend
  hold on;  % Keep the image, add patches for legend
  for i = 1:length(labels_y)
      patch("XData", [NaN, NaN], 
            "YData", [NaN, NaN], 
            "FaceColor", color_map(i, :), 
            "EdgeColor", "none");
  end
  legend(labels_y);
  hold off;  % Return to normal plotting behavior

  # Weighted Winner Classes - Part 9
  % Normalize h if not already softmax probabilities
  h_normalized = h ./ sum(h, 2);

  %% Preallocate an RGB image array filled with zeros
  % The size is the number of points in your grid, with 3 layers for R, G, B
  rgb_image_weighted = zeros(rows(feature1_grid), columns(feature1_grid), 3);

  %% For each class, add its color weighted by its probability to the image
  for i = 1:size(color_map, 1)
      % Extract the probability for class i and reshape it to match the grid
      class_probabilities = reshape(h_normalized(:, i), size(feature1_grid));
      
      % For each color channel, add the contribution of this class's 
      %color weighted by its probability
      for j = 1:3 % R, G, B channels
          rgb_image_weighted(:,:,j) += class_probabilities .* color_map(i, j);
      end
  end

  %% Visualize the weighted class color regions
  figure(fig_count++, "name", "Class Probability Weighted Regions");
  image(feature1_range, feature2_range, rgb_image_weighted); % Adjust axes if necessary
  set(gca, "YDir", "normal"); % Ensure the Y-axis direction is correct
  xlabel(sprintf("%s", labels_x{comb(index_min(1),1)}));
  ylabel(sprintf("%s", labels_x{comb(index_min(1),2)}));
  title("Class Probability Weighted Regions");

  % Adda legend
  hold on;  % Keep the image, add patches for legend
  for i = 1:length(labels_y)
      patch("XData", [NaN, NaN], 
            "YData", [NaN, NaN], 
            "FaceColor", color_map(i, :), 
            "EdgeColor", "none");
  end
  legend(labels_y);
  hold off;  % Return to normal plotting behavior


  %% The Same but polynomial matrix design - Part 10 
  printf("\n\n##### PART 10 #####\n\n");

  
  %% Plot the most important features
  
  %% Use the first combination with least error
  order = 12;
  Xtr_temp_poly = matrix_design(Xtr(:,comb(index_min(1),:)), order);
  Xte_temp_poly = matrix_design(Xtr(:,comb(index_min(1),:)), order);
  Theta0_poly=rand(columns(Xtr_temp_poly),columns(Ytr))-0.5; ## Random starting point
  try
    printf("Optimizing Theta for polynomial features\n");
    [ts,errs]=opt.minimize(@softmax_loss,Theta0_poly,Xtr_temp_poly,Ytr);
    Theta = ts{end};
  catch
    printf("\n### Error optimizing Theta polynomial \t");
    printf("\t: ###\n %s\n\n", lasterror.message);
  end_try_catch
  best_Theta = Theta;

  %% Plot the features
  feature1_range = linspace(min(Xte_temp(:,1)), max(Xte_temp(:,1)), 100);
  feature2_range = linspace(min(Xte_temp(:,2)), max(Xte_temp(:,2)), 100);
  [feature1_grid, feature2_grid] = meshgrid(feature1_range, feature2_range);

  %% Create the design matrix for the grid
  X = [feature1_grid(:), feature2_grid(:)];
  X_extended = matrix_design(X, order);

  %% Normalize the features
  %% Note: I can use the same normalizer nx but is the same cut the 
  %% features an then use a new normalizer and transform them. 
  %% It will generate the same results
  nx_temp_poly = normalizer(str_normalizer);
  Xtr_temp_poly_normal = nx_temp_poly.fit_transform(Xtr_temp_poly);
  Xte_temp_poly_normal = nx_temp_poly.transform(Xte_temp_poly);
  X_normal = nx_temp_poly.transform(X_extended);

  %% Calculate the probabilities
  h = softmax_hyp(best_Theta, X_normal);

  %% Reshape the probabilities to match the grid
  probabilities = zeros(rows(feature1_grid),columns(feature1_grid),columns(h));
  %% Plot the surfaces
  for i = 1:columns(h)
    probabilities(:,:,i) = reshape(h(:,i), size(feature1_grid));

    figure(fig_count++, "name", sprintf("Probability of Penguin being %s, Polynomial Matrix (order = %d)",
      labels_y{i}, order));
    surf(feature1_grid, feature2_grid, probabilities(:,:,i));
    xlabel(sprintf("%s", labels_x{comb(index_min(1),1)}));
    ylabel(sprintf("%s", labels_x{comb(index_min(1),2)}));
    zlabel(sprintf("p(y=%s|x)", labels_y{i}));
    title(sprintf("Probability of Penguin being %s, Polynomial Matrix (order = %d)",
     labels_y{i}, order));
    colorbar;
  endfor

  %% Plot the Winner Classes
  %% Assuming h is your matrix of class probabilities (points x classes)
  [~, class_indices] = max(h, [], 2);  % Find the class with the highest probability for each point

  %% Reshape to match the grid
  class_matrix = reshape(class_indices, size(feature1_grid));

  %% Create a color palette
  % This is just an example, adjust the number of colors based on your classes
  color_map = [1,0,0; 0,1,0; 0,0,1];

  %% Convert indices to RGB image
  rgb_image = ind2rgb(class_matrix, color_map);

  %% Visualize the class regions
  figure(fig_count++, "name", 
  sprintf("Winning Class Regions, Polynomial Matrix (order = %d)", order));
  image(feature1_range, feature2_range, rgb_image); % Adjust axes if necessary
  set(gca, "YDir", "normal"); % Ensure the Y-axis direction is correct
  xlabel(sprintf("%s", labels_x{comb(index_min(1),1)}));
  ylabel(sprintf("%s", labels_x{comb(index_min(1),2)}));
  title(sprintf("Winning Class Regions, Polynomial Matrix (order = %d)", order));



  % Adda legend
  hold on;  % Keep the image, add patches for legend
  for i = 1:length(labels_y)
      patch("XData", [NaN, NaN], 
            "YData", [NaN, NaN], 
            "FaceColor", color_map(i, :), 
            "EdgeColor", "none");
  end
  legend(labels_y);
  hold off;  % Return to normal plotting behavior

  # Weighted Winner Classes - Part 9
  % Normalize h if not already softmax probabilities
  h_normalized = h ./ sum(h, 2);

  %% Preallocate an RGB image array filled with zeros
  % The size is the number of points in your grid, with 3 layers for R, G, B
  rgb_image_weighted = zeros(rows(feature1_grid), columns(feature1_grid), 3);


  %% For each class, add its color weighted by its probability to the image
  for i = 1:size(color_map, 1)
      % Extract the probability for class i and reshape it to match the grid
      class_probabilities = reshape(h_normalized(:, i), size(feature1_grid));
      
      % For each color channel, add the contribution of this class's 
      %color weighted by its probability
      for j = 1:3 % R, G, B channels
          rgb_image_weighted(:,:,j) += class_probabilities .* color_map(i, j);
      end
  end

  %% Visualize the weighted class color regions
  figure(fig_count++, "name", 
  sprintf("Class Probability Weighted Regions, Polynomial Matrix (order = %d)", order));
  image(feature1_range, feature2_range, rgb_image_weighted); % Adjust axes if necessary
  set(gca, "YDir", "normal"); % Ensure the Y-axis direction is correct
  xlabel(sprintf("%s", labels_x{comb(index_min(1),1)}));
  ylabel(sprintf("%s", labels_x{comb(index_min(1),2)}));
  title(sprintf("Class Probability Weighted Regions, Polynomial Matrix (order = %d)", order));

  % Add legend
  hold on;  % Keep the image, add patches for legend
  for i = 1:length(labels_y)
      patch("XData", [NaN, NaN], 
            "YData", [NaN, NaN], 
            "FaceColor", color_map(i, :), 
            "EdgeColor", "none");
  end
  legend(labels_y);
  hold off;  % Return to normal plotting behavior

  
endif
