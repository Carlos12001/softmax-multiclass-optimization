## Copyright (C) 2022-2024 Pablo Alvarado
## EL5857 Aprendizaje Automático
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica


classdef normalizer < handle
  ## normalizer Normalization and de-normalization class.
  ##   This class stores the normalization factors for each column of
  ##   a design matrix, so that it can be applied later to any other data set.
  ##   Additionally it allow to denormalize data.
  ##
  ##   Normalize each column of a design matrix, according to the
  ##   specified normalization method:
  ##   "minmax": min value is mapped to -1 and max value is mapped to +1
  ##   "normal": mean is mapped to 0 and the stddev to 1
  ##
  ##   If a column of the matrix has no variance at all, it will be kept
  ##   untouched.

  ## -*- texinfo -*-
  ## @deftypefn{} fit(@var{X})
  ## Compute the normalization factors depending on @var{method} specified at
  ## construction time.
  ## @end deftypefn
  ##
  ## @deftypefn {} {@var{Xn}=} transform(@var{X})
  ## Take the given design matrix @var{X} and produce a new normalized version @var{Xn}




  properties
    ## Bias 
    bias = [];
    ## Slope
    slope = [];
    ## Normalization method in use
    method = "normal";
  endproperties
  
  methods
    function s=normalizer(meth="minmax")
      ## Construct a normalization object with the given normalization method.
      ## The method must be one of:
      ##   "minmax": min value is mapped to -1 and max value is mapped to +1
      ##   "normal": mean is mapped to 0 and the stddev to 1      

      s.method=meth;
    endfunction

    ## Compute the normalization factors
    ## If a column has zero variance, it is kept unchanged
    function fit(s,X)
      if (s.method=="minmax")
        mn=min(X);
        mx=max(X);
        dx=mx-mn;
        dy=2;
        dxx = (dx==0).*dy + dx;
        s.slope = dy./dxx;
        s.bias = (1 - s.slope .* mx).*(dx!=0);
      elseif (s.method=="normal")
        mn=mean(X);
        sd=std(X);

        stf = (sd==0) + sd;   ## avoid division by 0
        s.slope = 1./stf;
        s.bias = (-s.slope .* mn) .* (sd!=0);
      else
        error("Unknown normalization method");
      endif
    endfunction

    ## Transform the given data with the last computed
    ## factors
    function Xn=transform(s,X)
      Xn = s.slope.*X + s.bias;
    endfunction
    
    ## Compute the factors and transform the data
    function Xn=fit_transform(s,X)
      s.fit(X);
      Xn=s.transform(X);
    endfunction

    ## Inverse mapping: denormalize 
    function X=itransform(s,Xn)
      X = (Xn - s.bias) ./ s.slope;
    endfunction
    
  endmethods
endclassdef
