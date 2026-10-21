# Analytic curved Tet10 harmonic qualification

This Stage 4 envelope exercises genuinely non-affine Tet10 volume and Tri6
boundary geometry through physical assembly, load realization, direct harmonic
response and recovered complex displacement. It extends the independent
[straight rod reference](solid-rod-harmonic.md), not production CAD generation or
general curved-solid qualification. No stage or qualification gate is promoted by
this fixture alone.

## Geometry and independent reference

Map the unit cube with the fixed quadratic map

```
F(x,y,z) = (x, y + eta*z*(1-z), z), eta = 0.4
det(DF) = 1
```

Every corner and midside node is transformed; the quadratic interpolation
therefore represents this map exactly. An edge with endpoint difference `dz`
has midside bow `eta*dz^2/4`, so this is not the affine Tet10 path. Canonical
global-edge midsides and complete Tri6 connectivity are retained. Exact UV
parameters denote preimage coordinates; chart identities are resealed against
the analytic transformed evaluator. This is analytic fixture provenance, not an
OCCT evaluator or meshing-convergence claim.

Independent volume is 1 m³, volume centroid is `(1/2,1/2+eta/6,1/2)` m, and each
end has area 1 m² with that same transverse centroid. The two curved side areas
are each

```
A_side = (sqrt(1+eta^2) + asinh(eta)/eta)/2
A_total = 4 + 2*A_side
```

The closed exterior area-vector sum is zero and every lateral surface has
`n_x=0`. With `nu=0`, constrained transverse motion and displacement
`u=(U(x),0,0)`, the same axial continuum PDE remains exact on this curved
extrusion. No slenderness approximation, surrogate coarse mesh or RunMat
operator supplies the reference.

For `E=100 Pa`, `rho=1 kg/m³`, `T=1+0.4i Pa`, and `exp(+i omega t)`, the undamped
solution is

```
c = 10 m/s, k = omega/c
U(x) = T*sin(k*x)/(E*k*cos(k))
R_left = -T/cos(k)
```

The damped reference uses `alpha=0.8 s^-1`, `beta=0.002 s`:

```
Ec = E*(1+i*omega*beta)
q^2 = rho*(omega^2-i*omega*alpha)/Ec
U(x) = T*sin(q*x)/(Ec*q*cos(q))
R_left = -T/cos(q)
P_in = omega/2 * Im(T*conj(U(1)))
P_loss = omega^2/2 * integral[alpha*rho*|U|^2 + beta*E*|U'|^2] dx
```

Entire sine/cosine series in `q²` avoid square-root branch ambiguity. Independent
1024-interval Simpson integration checks continuum loss against boundary power.
The constitutive derivation and source references are in the straight-rod
qualification document.

## Mandatory checks and bounds

All cases use 1/2/4 cells per axis with the same fixed curvature. Refinement does
not flatten the domain or discard high-order nodes. Geometry checks include
volume/first moments, end area/centroid, curved exterior area, closed area-vector
balance, positive integration weights and a physical affine axial strain/stress
patch using the shared curved inverse-Jacobian recovery path.

Undamped `k={0.5,1,2}` covers both sides of the first resonance. Each load phasor
has independent resultant `(T,0,0)` and origin moment
`(0,T/2,-T*(1/2+eta/6))`. Nodal complex-field and reaction errors must decrease by
more than six per refinement and finish below 0.2%. Algebraic residual is at
most `1e-10`, physical balance error below `1e-8`, and undamped mean input power
below `1e-9 W`.

Damped `k={0.5,pi/2,2}` includes the undamped resonance. Complex-field error must
decrease by more than five per refinement; finest field/reaction errors must be
below 0.3%. Input and FE loss must be positive and agree within `1e-8` relative;
finest input power must agree with the independent continuum within 0.5%.
Continuum volume loss and boundary power agree within `1e-9` relative. The
explicit 256-vector/8192-iteration policy retains the `1e-10` residual target;
ordering and resource proof ceilings are unchanged.

The cold case serializes/decodes the workload, canonical mesh, entry request,
result and recovered complex displacement. Both undamped and damped 4-cell
cases must satisfy the independent continuum phase bound, not merely match a
hot result. A separately valid mesh with `eta=0.41` and consistently changed
chart identities, and a changed traction, must reject reuse of the original
admitted request.

```sh
cargo test --locked -p runmat-analysis-fea --lib pipeline::modal::rod_qualification::curved -- --nocapture
```

## Evidence and remaining gates

Geometry/recovery, undamped and damped continuum tests pass with the original
bounds. Finest undamped relative field errors at `k={0.5,1,2}` are
`0.000024092`, `0.000090141`, `0.000406960`; corresponding reaction errors are
`0.00000004075`, `0.00000369865`, `0.000164745`. Finest damped field errors at
`k={0.5,pi/2,2}` are `0.0000243995`, `0.000411943`, `0.000407019`.
The focused suite passes all four tests, including cold execution/recovery for
both damping settings and consistent changed-curvature rejection. The four
existing straight Tet4/Tet10 cold execution/recovery tests also pass against the
same compiled test binary, as does the original both-order damped continuum
helper with its unchanged cases and bounds. These are portable-object execution checks, not actual
process replacement or a complete qualification gate decision.

This envelope does not establish curved CAD generation, arbitrary deformation
maps, bending/shear spectra, nonzero-Poisson unrestrained rods, prestress, real
process failure/replacement, measured peak resources, GPU/distributed algebra,
or complete accuracy/release admission. The required damped-modal Tet4 proof
limit remains an explicit unresolved Stage 4 gate.
