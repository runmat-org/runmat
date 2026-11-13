"""P0 floor of Tet4 stress: python3 floor.py tet4 4 (moving_front/verification.rs)."""
import sys, numpy as np
from j2front import *
quad = sys.argv[1]=="tet10"; cells=int(sys.argv[2])
out,(X,tets,pts,u,al,ep) = run(cells, quad, [i/8 for i in range(1,9)], verbose=False)
# recompute per factor floors: P0 projection floor of reference stress at qp
for f in [0.125,0.25,0.375,0.5,1.0]:
    by = {}
    for (e,N,G,w,x) in pts:
        by.setdefault(e,[]).append((w,reference(x[0],f)[3]))
    err=nrm=0
    for e,l in by.items():
        W=sum(w for w,_ in l); m=sum(w*s for w,s in l)/W
        for w,s in l: err+=w*((s-m)**2).sum(); nrm+=w*(s**2).sum()
    print(f, "P0 floor", np.sqrt(err/nrm))
