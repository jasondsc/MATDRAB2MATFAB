function [dprime,criterion] = dprime(pHit,pFA,nSignal,nNoise)
% Calculate sensitivity index from signal-detection theory
%
%  USE:
%  dvalue = dprime(pHit,pF, nSignal, nNoiseA) returns the sensitivity index, using a
%  standard value for correction of rates of 0 and 1.
%  dvalue = dprime(pHit,pFA,nTarget,nDistract) uses the given numbers of
%  targets and distractor trials value for correction.
%  [dvalue,cvalue] = dprime(pHit,pFA) returns also the individual bias
%  criterion.
%
%  Coding by Martin Böckmann-Barthel, Otto-von-Guericke-Universität Magdeburg
%  The sensitivity index d' measures the distance between the signal and
%  the noise means in standard deviation units. c is the distance of the
%  bias criterion %  from the point where neither response is favored, also
%  in standard units. Positive c values indicate a bias towards high hit
%  rates and false alarm rates
% 
%  Obligatory input variables: 
%  pHit, pFA - hit rate and false alarm rate (max: 1.0, min 0.0)
%  Optional variables: nTarget, nDistract - number of targets and
%  distractor trials (needed for correction of perfect responses)
%  Perfect hit rates (pHit = 1) and FA rates (pFA = 0) are corrected 
%  by -1/(2 nTarget) and +1/(2 nDistract), respectively, if provided
%  cf. Stanislaw H, Todorov N, Behav Res Meth (1999) 31, 137-149, "1/2N rule"

%% Check if input is correct 
if any(pHit > 1 | pFA > 1 | pHit < 0 | pFA < 0 )
    error('Please enter probabilities (i.e., values between 0 and 1)!)');
end 

if length(pHit)== length(pFA) == length(nSignal) == length(nNoise)
    error('Please make sure all inputs are of the same length');
end 

%% correct for zeros 

correctphit=find(pHit==1);
if size(correctphit) >=1
    pHit(correctphit)= (2*nSignal(correctphit)-1)./(2*nSignal(correctphit));
end
correctphit=find(pHit==0);
if size(correctphit) >=1
    pHit(correctphit)= 1./(2*nSignal(correctphit));
end

correctpfa=find(pFA==1);
if size(correctpfa) >=1
    pFA(correctpfa)= (2*nNoise(correctpfa)-1)./(2*nNoise(correctpfa));
end
correctpfa=find(pFA==0);
if size(correctpfa) >=1
    pFA(correctpfa)= 1./(2*nNoise(correctpfa));
end


%% Convert values to z-scores
zHit = -sqrt(2).*erfcinv(2*pHit);
zFA = -sqrt(2).*erfcinv(2*pFA);


%% Calculate Sensitivity and cirteria
dprime = zHit - zFA ;
if nargout > 1
    criterion = -.5*(zHit + zFA);
end 
