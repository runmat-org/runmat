# Thermal surface-to-ambient radiation — bounded face-form evidence

Status: experimental local operator and authored contract. **Not an admitted thermal solve.**
Plan ownership: Stage 5 radiation boundaries, nonlinear closure and heat accounting. The
canonical thermal assembler deliberately rejects `ThermalRadiation` until the whole nonlinear
procedure is wired. No Stage 5, restart, resource or production gate is promoted by these tests.

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
this does not yet establish whole-procedure peak memory or backend admission.

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

## Required continuation / gate decision

Keep admission closed until damped Newton steady and backward-Euler execution, positivity,
nullspace/anchor handling, residual-based nonlinear acceptance, physical boundary heat accounting,
source/signal equivalence, resource reservations and backend provenance are implemented together.
Then require spatial/time convergence and independent equilibrium/cooling references, hot/cold
workload execution, invalid-source/geometry tests, solver-specific restart and accuracy evidence.
Temperature-dependent material coefficients remain a separate real Stage 5 requirement.
Enclosure/view factors and participating-medium radiation are outside this bounded ambient law.
