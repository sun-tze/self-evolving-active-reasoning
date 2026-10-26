from math import comb
import random

def quantile(a,p):
    x=(len(a)-1)*p;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def cp(k,n,alpha=.025):
    def cdf(k,p):return sum(comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
    def inverse(k,target):
        lo,hi=0.,1.
        for _ in range(60):
            mid=(lo+hi)/2
            if cdf(k,mid)>target:lo=mid
            else:hi=mid
        return (lo+hi)/2
    return [0 if k==0 else inverse(k-1,1-alpha/2),1 if k==n else inverse(k,alpha/2)]
def paired(a,b):
    d=[int(x)-int(y) for x,y in zip(a,b)];n=len(d);rng=random.Random(62026)
    boot=sorted(sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    w=sum(x==1 for x in d);l=sum(x==-1 for x in d);m=w+l
    wi,li=cp(w,n),cp(l,n)
    return dict(delta=sum(d)/n,wins=w,losses=l,ties=n-m,bootstrap_CI95=[quantile(boot,.025),quantile(boot,.975)],
        conservative_simultaneous_binomial_CI95=[wi[0]-li[1],wi[1]-li[0]],
        paired_exact_two_sided_p=min(1.,2*sum(comb(m,k) for k in range(min(w,l)+1))/2**m) if m else 1.,N=n,
        sparse_note='Paired percentile bootstrap can degenerate at the floor; simultaneous exact binomial bounds remain nondegenerate. No null/equivalence inference.')
