% subgradient of function y = log(rx^0.5 + C)/log(r+1)

function [S] = Lin(S,r)

C = 1; % Large value of C(such as 500) will improve the precision of MC;

S = r*0.5./(r*S + C*S.^0.5);