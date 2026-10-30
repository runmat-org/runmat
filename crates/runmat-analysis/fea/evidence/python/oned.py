# 1D P2 Galerkin (moving_front/verification.rs front derivation) for the same uniaxial-strain J2 bar (monotone loading -> path independent):
# sigma(eps) = M eps (elastic) or T eps + OFF (plastic). Exact Gauss-5 per element plus exact
# split at the kink is NOT used: plain 5-point Gauss, so this mirrors a standard element.
import numpy as np, j2front as J
f=0.375; M=J.M; T=J.T; OFF=J.OFF; ey=7.5/M
def sig(e): return (M*e, M) if M*e<=7.5 else (T*e+OFF, T)
g,w=np.polynomial.legendre.leggauss(5)
def solve(n):
    nn=2*n+1; x=np.linspace(0,1,nn); u=np.zeros(nn)
    for it in range(60):
        K=np.zeros((nn,nn)); R=np.zeros(nn)
        for e in range(n):
            idx=[2*e,2*e+1,2*e+2]; a=x[2*e]; h=1/n
            for s,ww in zip((g+1)/2,w/2):
                N=np.array([(1-s)*(1-2*s),4*s*(1-s),s*(2*s-1)]); dN=np.array([4*s-3,4-8*s,4*s-1])/h
                st,tg=sig(dN@u[idx]); R[idx]+=dN*st*ww*h - N*f*(-T*J.B)*ww*h; K[np.ix_(idx,idx)]+=tg*np.outer(dN,dN)*ww*h
        R[-1]-=f*J.stress_full(1.0)[0]
        if np.linalg.norm(R[1:])<1e-14: break
        u[1:]-=np.linalg.solve(K[1:,1:],R[1:])
    err=nrm=0
    for e in range(n):
        idx=[2*e,2*e+1,2*e+2]; h=1/n
        for s,ww in zip((g+1)/2,w/2):
            dN=np.array([4*s-3,4-8*s,4*s-1])/h; eps=dN@u[idx]; xx=x[2*e]+s*h
            al=max((2*J.mu*eps-J.Y)/(3*J.mu+J.H),0) if M*eps>7.5 else 0.0
            err+=ww*h*(al-J.reference(xx,f)[2])**2; nrm+=ww*h*J.alpha_full(xx)**2
    return np.sqrt(err/nrm)
print("front r =", J.front(f))
for n in [1,2,4,8,16,32,64,128,256]: print(n, "h/r=%.2f"%(1/n/J.front(f)), "alpha_err=%.4e"%solve(n))
