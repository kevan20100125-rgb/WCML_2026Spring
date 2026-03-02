% Generate random test discrete probability density (pdf) P and Q
n = 5;
P=rand(1,n); P=P/sum(P);
Q=rand(1,n); Q=Q/sum(Q);
% Compute TVD using definition
n = length(P);
b = logical(dec2bin(0:2^n-1)-'0');
d = zeros(1,size(b,1));
for k=1:size(b,1)
    bk = b(k,:);
    Pa = sum(P(bk));
    Qa = sum(Q(bk));
    dPaQa = abs(Pa-Qa);
    d(k)= dPaQa;
end
dPQ = max(d)   ;

% Compute TVD using L1-norm
dFormula = 0.5 * norm(P-Q,1);