# Canonical acoustic harmonic qualification

## Current envelope and owner

`physics::acoustic::solve_canonical_harmonic` owns a CPU library solve for one explicit
acoustic harmonic step and positive frequency on a sealed canonical Tet4/Tet10 mesh.
The shared property compiler resolves each element's material; there is no material
averaging, synthetic frequency, generated topology, artificial damping or inferred boundary.
Sparse operators use the shared algebra and matrix-free GMRES implementation.

Supported fluid laws are lossless isotropic density and sound speed. Essential pressure
values and normal velocity use the shared typed signal realization, including complex
frequency ordinates and step-selected excitation cases. Velocity is **outward** from the
fluid. Normal-velocity sources are uniform kinematic values on exact exterior faces and
carry shared scalar-integral conservation evidence. Pressure sources are essential
boundary values, not a body force or pressure-to-load conversion. Unspecified exterior
faces have the homogeneous natural rigid-wall condition. Explicit boundaries support
rigid walls and positive real specific impedance. Prescribed pressure uses affine lifting.

Unsupported radiation, periodic/Bloch constraints, incident sources, material attenuation,
nonuniform distributions, interior boundary selection and structural/coupled models fail
closed. This entry has no production admission, native study integration, resource envelope,
restart or GPU/distributed claim. The public/runtime graph route remains a separately
identified experimental proxy until its callers can migrate with qualification evidence.

## Equations and independent references

With peak phasors and `exp(+i omega t)`, fluid momentum gives
`v = i grad(p)/(omega rho)`. The Helmholtz weak operator is

$$
(K-\omega^2 M+i\omega B)p=b,\quad
K_{ij}=\int_\Omega\rho^{-1}\nabla N_i\cdot\nabla N_j\,dV,\quad
M_{ij}=\int_\Omega(\rho c^2)^{-1}N_iN_j\,dV.
$$

For specific impedance `Z = p/v_n`, `B_ij = integral_boundary N_i N_j/Z`.
An imposed outward velocity supplies `b_i = -i omega integral_boundary N_i v_n`.
These signs follow the frequency-domain fluid equations described in
[COMSOL's pressure-acoustics theory](https://doc.comsol.com/6.3/doc/com.comsol.help.aco/aco_ug_pressure.05.164.html).
The references below are analytic solutions of those equations, not another RunMat solve.

The unit Tet4 reference uses `rho=3 kg/m³`, `c=2 m/s`, volume `1/6 m³`, exact
constant barycentric gradients, `integral N_i²=1/60 m³`, and
`integral N_i N_j=1/120 m³` for distinct nodes. Every K/M entry is checked.
Straight and edge-warped Tet10 cases check partition of unity, total compressibility,
and affine pressure-gradient/particle-velocity recovery in physical coordinates.

The unit-area, unit-length duct uses the same fluid and `f=0.2 Hz`. For matched
termination `Z=rho c=6 Pa s/m`, the exact pressure is

$$
p(x)=(1+0.4i)e^{-ikx},\quad v_x=p/(\rho c),\quad k=2\pi f/c.
$$

Two drives are checked independently: prescribed complex pressure at `x=0`, and
outward velocity `v_n=-(1+0.4i)/6 m/s` at `x=0`. Exact absorbed mean power is
`|1+0.4i|²/(2 rho c)=1.16/12 W`. Uniform canonical tetrahedral cubes with
2, 4 and 8 cells per axis establish pressure amplitude/phase convergence; every
refinement must reduce nodal RMS error by more than a factor of three. Finest RMS
pressure error must be below `0.004 Pa`; finest recovered velocity RMS error must
be below `0.01 m/s`. Source/termination mean power must close within `1e-8` of
apparent power, and termination loss must approach the independent analytic value.

For a rigid right termination and unit left pressure, the independent standing wave
is `p(x)=cos(k(1-x))/cos(k)`. The same refinement sequence verifies pressure
convergence and no artificial mean loss. Reactive apparent power normalizes balance,
so a lossless case does not divide by zero real power or use a dimensional floor.

### Quadratic continuum and forced rigid cavity

The straight Tet10 unit-cube fixture shares midsides by canonical global-edge identity,
uses Tri6 faces with actual planar source-face parameters, and is sealed/validated through
the ordinary meshing contract. It is a qualification fixture, not a generated replacement
for imported production topology. At `f=0.2` and `0.5 Hz`, the velocity-driven matched
wave uses 1/2/4 cells per axis. Nodal relative RMS pressure error must decrease by more
than six on each refinement and finish below 0.4 percent. Finest particle-velocity RMS
error must be below `0.01 m/s`, intensity RMS error below `0.005 W/m²`, and energy-density
RMS error below `0.005 J/m³`. Residual and boundary-power tests are separate from these
discretization-accuracy tests.

For the same cube with rigid side/right walls and prescribed outward left velocity
`v_n=-(1+0.4i)/6 m/s`, the independent axial forced-cavity solution is

$$
p(x)=i\rho c v_n\frac{\cos(k(1-x))}{\sin k},\qquad
v_x(x)=-v_n\frac{\sin(k(1-x))}{\sin k}.
$$

The axial poles are `f_n=n c/(2L)=n Hz`. Tests at 0.9/1.1 Hz check convergence on
both sides of the first pole and the reversal of boundary pressure phase. Tet4 uses
4/8/16 cells per axis, Tet10 1/2/4. Finest pressure relative RMS error must be below
10 percent; energy-density relative RMS error below 15 percent. The eight-cell Tet4
case failed the energy target above resonance, motivating finer resolution rather
than a relaxed target. These are bounded forced-response checks, **not** an acoustic
eigenvalue solver, exact-pole uniqueness test, or arbitrary resonance-conditioning certificate.

### Intensity and energy recovery

Measured finest-mesh nodal relative RMS pressure errors for these bounded cases:

| Case | Frequency (Hz) | Element | Error (%) |
|---|---:|---|---:|
| Matched traveling wave | 0.2 | Tet10 | 0.001377 |
| Matched traveling wave | 0.5 | Tet10 | 0.021076 |
| Forced rigid cavity | 0.9 | Tet4 | 1.076073 |
| Forced rigid cavity | 1.1 | Tet4 | 2.266279 |
| Forced rigid cavity | 0.9 | Tet10 | 0.188602 |
| Forced rigid cavity | 1.1 | Tet10 | 0.376441 |

Canonical recovery now reports volume-averaged particle velocity, time-averaged intensity
`I=Re(p conjugate(v))/2`, and lossless acoustic energy density
`E=(|p|²/(rho c²)+rho |v|²)/4`, using peak phasors. Products are evaluated at physical
quadrature points before volume averaging, not reconstructed from averaged fields.
The intensity convention follows
[COMSOL's frequency-domain postprocessing equations](https://doc.comsol.com/6.3/doc/com.comsol.help.aco/aco_ug_pressure.05.140.html);
the energy expression follows averaging the pressure/compression and fluid kinetic energies.
No COMSOL numerical output is used as evidence.

The traveling wave has `I=(1.16/12,0,0) W/m²` and `E=1.16/24 J/m³`.
The forced rigid cavity has zero mean intensity/loss and constant
`E=1.16/(48 sin² k) J/m³`. A separate straight Tet10 polynomial reference,
`p=x²+i x`, checks `I_x=1/60 W/m²` and `E=401/3360 J/m³` using exact
tetrahedral moments; multiplying averaged pressure and velocity would wrongly give
`I_x=1/240`. An affine straight-tetrahedron reference checks `E=119/48`.
Curved Tet10 retains the earlier integral/affine-velocity evidence; this does not qualify
curved non-affine energy-form integration or all geometric/quadrature accuracy envelopes.

## Verification and remaining gates

```sh
cargo test --locked -p runmat-analysis-fea --lib physics::acoustic::canonical
cargo test --locked -p runmat-analysis-fea --lib
cargo clippy --locked -p runmat-analysis-fea --all-targets --no-deps -- -D warnings
```

The executable cases also reject missing faces, overlapping explicit boundaries,
zero frequency, material damping and unsupported radiation. Field recovery returns
nodal complex pressure and volume-averaged element velocity, with immutable mesh
projection identity, source evidence and power metrics.

Stage 8 remains open. Required follow-on work includes wider Tet10/curved continuum envelopes,
heterogeneous interfaces, cavity eigenfrequencies and resonance/uniqueness diagnostics,
radiation references, complex/frequency-dependent impedance where advertised,
periodic/Bloch constraints, incident-field support, explicit/generated spectrum
equivalence, prepared resource/accuracy reports, authored/native workload integration,
observations/storage and cold restart, parallel frequency execution and blocking release
evidence. These library tests do not qualify those gates or remove the graph route's
remaining caller migration obligation.
