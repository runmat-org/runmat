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

Stage 8 remains open. Required follow-on work includes Tet10 continuum convergence,
heterogeneous interfaces, cavity eigenfrequencies and resonance/uniqueness diagnostics,
radiation references, complex/frequency-dependent impedance where advertised,
periodic/Bloch constraints, incident-field support, explicit/generated spectrum
equivalence, prepared resource/accuracy reports, authored/native workload integration,
observations/storage and cold restart, parallel frequency execution and blocking release
evidence. These library tests do not qualify those gates or remove the graph route's
remaining caller migration obligation.
