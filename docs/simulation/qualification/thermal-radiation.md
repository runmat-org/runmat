# Thermal surface-to-ambient radiation — bounded nonlinear procedure

Status: CPU FP64 canonical Tet4/Tet10 steady and backward-Euler execution.
Plan ownership: Stage 5 radiation boundaries, nonlinear closure and heat accounting.
Linear thermal operator entry points still reject radiation explicitly; the nonlinear route
retains the actual boundary law and Jacobian. This bounded envelope does not complete Stage5
or qualify GPU, distributed execution, numerical process restart or release.

## Law and independent references

Outward heat loss is `q = epsilon sigma (T^4 - Ta^4)`; its derivative is
`dq/dT = 4 epsilon sigma T^3`. This reverses the inward-flux sign used in
[COMSOL's surface-to-ambient radiation documentation](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_heattransfer.30.34.html).
The constant `sigma = 5.670374419e-8 W/(m^2 K^4)` is the rounded
[NIST/CODATA SI value](https://physics.nist.gov/cuu/pdf/JPCRD2022CODATA.pdf).
These are law/constant references, not a claim of comparison against COMSOL numerical output.

Face residuals are `integral Ni q dA`; tangents are `integral Ni Nj dq/dT dA`.
The implementation factors the fourth-power difference near equilibrium rather than
subtracting two large fourth powers. Kelvin, emissivity and overflow checks fail closed;
zero emissivity means exact insulation, not an invented positive Robin anchor.

## Integration and checks

The default degree-five Dunavant rule remains unchanged for existing consumers. The explicit
shared face integration rule uses 8x8/12x12 Duffy-Gauss products, refining to 16x16 when required.
Tri6 radiation forms can reach degree ten even on flat faces. Curved faces require convergence
of the **actual residual and tangent**, not just geometry moments, at relative tolerance 1e-10
scaled by physical transfer and integrated tangent. Exact zero transfers have no ambient-T floor.
Whole-triangle quadratic orientation checks remain mandatory. Scratch is fixed face-local arrays;
this does not establish measured whole-process peak memory. Numerical admission conservatively
reserves 10 simultaneous full-support CSR payloads, a diagonal damping payload and 24N double
scratch values, plus the existing masks/component traversal, fields, source/solver metadata,
nonlinear evidence and serialized result profiles. Original resident ceilings are unchanged.

Tests cover:

- outward/inward signs, equilibrium, zero emissivity, zero Kelvin, invalid ranges and overflow;
- near-equilibrium agreement with the differential limit;
- constant Tri3/Tri6 power conservation, symmetric tangent and finite-difference consistency;
- quadratic `T=300+100*x^2` on the unit right triangle against binomial expansion using
  `integral x^n dA = 1/((n+1)(n+2))`; the old degree-five rule demonstrably underintegrates;
- a quadratic graph `z=a*x^2`, `a=0.2`, against independently integrated surface area
  `0.5*(sqrt(1+b^2)+asinh(b)/b)-((1+b^2)^(3/2)-1)/(3*b^2)`, `b=2*a`;
- negative interpolated Tri6 Kelvin despite nonnegative nodal Kelvin;
- authored lowering, dimensionless emissivity, model validation and boundary wire roundtrip.

## Whole-procedure acceptance and independent witnesses

The shared sparse conductivity/capacity operators are reused. Newton uses the original
free-coordinate residual, consistent radiation tangent, exact prescribed increments and
typed invalid-Kelvin backtracking. At most 32 corrections and 24 trials per correction are
permitted. Acceptance requires equilibrium, accepted departure-coordinate increment and
original physical heat balance all at the unchanged workload target. Inner PCG residuals
are retained separately; they are not nonlinear acceptance. No absolute-temperature or
dimensioned power floor creates an artificial convergence claim.

Actual nonlinear sample records enter result metadata and numerical identity. Receipts
bind their presence to the original boundary law and retain maximum equilibrium/increment
errors. Distinct steady/BE damped-Newton formulation identities prevent a linear-solve label.
Nonlinear `GlobalEquilibrium/ResidualBound` accuracy evidence uses the original nonlinear
residual, not the last linear correction. Semantic-state publication binds the exact retained
result identity; it does not recalculate or substitute a lost temperature field.

On a unit-length/unit-area slab, `k=2`, hot face 400 K, ambient 300 K and emissivity 0.7,
independent monotone bisection solves `k(400-Ts)=epsilon*sigma*(Ts^4-300^4)`.
Both element orders reproduce the exact linear temperature, gradient, flux, radiation loss
and prescribed-boundary power. A 1 K initial reference exercises genuine backtracking.
Invalid Kelvin, insufficient original resources and GPU radiation requests reject execution.

The manufactured continuum is `T(x,t)=400-10x-20t^2`, `rho*c=1`, `k=2`, over 0–1 s.
Generation is `-40t`, left temperature is prescribed, sides insulated and right incoming
flux is `epsilon*sigma*(T(1,t)^4-300^4)-20`. Original tabulated values are exact at solve
knots; no false interpolation-error certificate is supplied. A real prior linear steady
producer supplies the spatial initial temperature through the shared retained-field artifact
and canonical initial-field binding, not point-target admission or fabricated solution data.

Maximum errors over all applicable snapshots at 8/16/32 intervals:

| Order | Temperature, K | Gradient/flux, relative | Storage-rate error divided by40 W/m³ |
| --- | --- | --- | --- |
| Tet4 | .225594 / .112780 / .0563825 | .0451187 / .0225560 / .0112765 | .0624221 / .0312189 / .0156116 |
| Tet10 | .217295 / .108646 / .0543175 | .0434362 / .0217177 / .0108576 | .0623301 / .0311816 / .0155952 |

Each field must improve by more than 1.8 per halving of the timestep; the original finest
temperature bound remains 0.1 K. These are independent temporal checks on fixed spatial meshes,
not a spatial-convergence certificate or a new absolute flux/storage accuracy bound.
All accepted samples also satisfy the unchanged 1e-9 nonlinear/linear/physical-power target.
Canonical manifest/receipt/state roundtrips retain source lineage; missing/excess nonlinear
evidence rejects publication, and accepted-record tampering changes numerical identity and
invalidates the original receipt/state request. These are portable object roundtrips, not a
failed-process restart or remote worker witness.

## Required continuation / gate decision

Qualification remains open. Required continuation includes broader independent spatial and
curved whole-procedure references, authored/generated signal equivalence, actual worker/cold
process execution, solver-specific restart, satisfied full accuracy policies, measured usage
and blocking release evidence. Existing face-form curved/quadrature tests do not prove those.
Temperature-dependent material coefficients remain a separate real Stage 5 requirement.
Enclosure/view factors and participating-medium radiation are outside this bounded ambient law.
