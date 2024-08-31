% ===============================
% Self-adaptive penalty function
% ===============================

function f=SaPenalty(val,tcons,pop)

[NP,D]=size(pop);
[fsort,temp]=sort(val);
if(fsort(NP)==fsort(1))
    fnorm=ones(NP,1);
else
fnorm=(val-fsort(1))./(fsort(NP)-fsort(1));
end

tcons_max=max(tcons);

tcons_min=min(tcons);

if(~(tcons_max==0))
    tcons=tcons/tcons_max;
end

f_index=find(tcons==0);
rf=length(f_index)/NP;
if rf==0,
    X=zeros(NP,1);
    d=tcons;
else
    X=tcons;
    d=sqrt(fnorm.^2+tcons.^2);
end

Y=fnorm;
Y(f_index)=zeros(length(f_index),1);

p=(1-rf).*X + (rf.*Y);

f=d+p;
