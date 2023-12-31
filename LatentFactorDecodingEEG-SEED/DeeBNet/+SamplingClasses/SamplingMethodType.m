%****************************In the Name of God****************************
%SamplingMethodType class is an enumeration that contains types of sampling
%methods is used in RBM.
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our web page.
%
% The programs and documents are distributed without any warranty, express
% or implied.  As the programs were written for research purposes only,
% they have not been tested to the degree that would be advisable in any
% important application.  All use of these programs is entirely at the
% user's own risk.

% CONTRIBUTORS
%	Created by:
%   	Mohammad Ali Keyvanrad (http://ceit.aut.ac.ir/~keyvanrad)
%   	12/2015
%           LIMP(Laboratory for Intelligent Multimedia Processing),
%           AUT(Amirkabir University of Technology), Tehran, Iran
%**************************************************************************
%SamplingMethodType class or enumeration
classdef SamplingMethodType
   % Gibbs: Gibbs sampling method
   % CD: Contrastive Divergence sampling method
   % PCD: Persistent Contrastive Divergence sampling method
   % FEPCD: Free Energy in Persistent Contrastive Divergence sampling method
    properties (Constant)
      Gibbs=1;
      CD=2;
      PCD=3;
      FEPCD=4;
    end
    
end %End Classdef

