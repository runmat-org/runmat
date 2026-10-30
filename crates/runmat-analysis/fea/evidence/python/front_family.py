"""Tet10 axial-bar family monotonicity: python3 front_family.py tet10 36,70 1 (moving_front/verification.rs)."""
import sys, time, j2front as J
lat=int(sys.argv[3]) if len(sys.argv) > 3 else 3; seq=[int(c) for c in sys.argv[2].split(",")]; quad=sys.argv[1]=="tet10"
runs=[]
for n in seq:
    t=time.time(); out,_=J.run(n, quad, [i/8 for i in range(1,9)], verbose=False, lateral=lat); runs.append(out)
ok=True
for i in range(8):
    for k in ['disp','stress','alpha']:
        v=[r[i][k] for r in runs]
        good=all(b<a or (a<1e-10 and b<1e-10) for a,b in zip(v,v[1:]))
        if not good: ok=False; print("NONMONO inc",i+1,k,v)
print("all monotone" if ok else "fail")
for n, r in zip(seq, runs):
    print(n, " ".join("%.4e/%.4e/%.4e" % (o['disp'], o['stress'], o['alpha']) for o in r))
