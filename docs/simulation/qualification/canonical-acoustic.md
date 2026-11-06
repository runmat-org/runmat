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
rigid walls, positive real specific impedance and first-order local outgoing radiation.
Prescribed pressure uses affine lifting. Radiation uses the unique exterior-face owner's
compiled density and sound speed, not a global material. Independent planar
normal-incidence evidence and its still-failed convergence gate are described in
[local radiation](acoustic-local-radiation.md); radiation is not yet qualified.

Matching translated exterior-face periodic/Bloch constraints use unit-phase pressure
relations; rotated and nonmatching pairing remains unsupported.

Unsupported higher-order/exterior-domain radiation, incident sources, material attenuation,
nonuniform distributions, interior boundary selection and structural/coupled models fail
closed. Stage 8 qualification, process restart and GPU/distributed execution remain
open. Authored canonical study execution and shared complex field publication
are connected; the bounded eight-case native matched-duct matrix, generated/explicit
equivalence and authenticated cold fields are retained in
[the right-LU qualification](acoustic-right-lu.md). This does not qualify every
boundary, source, frequency schedule or legacy graph entry point.

## Equations and independent references

With peak phasors and `exp(+i omega t)`, fluid momentum gives
`v = i grad(p)/(omega rho)`. The Helmholtz weak operator is

$$
(K-\omega^2 M+i\omega B)p=b,\quad
K_{ij}=\int_\Omega\rho^{-1}\nabla N_i\cdot\nabla N_j\,dV,\quad
M_{ij}=\int_\Omega(\rho c^2)^{-1}N_iN_j\,dV.
$$

For specific impedance `Z = p/v_n`, `B_ij = integral_boundary N_i N_j/Z`.
Local outgoing radiation uses `Z=rho*c`. The retained `mean_impedance_loss_w`
metric is total passive boundary-admittance power, including radiation; it is not
restricted to explicitly authored impedance boundaries.
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
for imported CAD topology. At `f=0.2` and `0.5 Hz`, the velocity-driven matched
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

### Portable numerical execution

`execution::AcousticHarmonicWorkload` binds the authored model, explicit scheduled
frequency, canonical formulation identity, Krylov policy and power-balance target.
The canonical Tet4/Tet10 realization has its own Galerkin operator, temporal, recovery
and solver identities; it cannot be confused with the legacy domain-graph method.

Its portable result retains nodal complex pressure, element velocity, intensity and
energy, ordered topology IDs, residual/power metrics and source-transfer evidence.
Canonical decoding validates shapes, finite physical values and accepted numerical
targets. Import against a workload and sealed mesh checks exact workload, formulation,
mesh and projection lineage; changed sources or valid resealed geometry are rejected.
This is compatibility validation, not proof that arbitrary imported field values solve
the equations. Independent traveling-wave checks cover cold object round-trips on
both Tet4 and Tet10.

Complete authored acoustic-complex output declarations remain part of workload
identity. The numerical result retains exact producer and publication-declaration lineage;
shared state publication is a separate admitted task, not part of the numerical solve.
This does not establish process restart, native authoring integration or graph-caller migration.

### Authored product and native field/state publication

The new core product path compiles a `RealizedStudy` against exact admitted geometry,
resolved persistent regions, materials and a sealed canonical mesh, then derives its
frequency entry through the shared `FrequencyResponsePlan`. The initial envelope is
one fresh direct harmonic step, one frequency and one complete `AcousticComplex`
declaration. Observations, authored output bindings, inbound state, graph sequencing
and multi-frequency state ambiguity reject explicitly. The model's required typed
state declaration is preserved, not dropped to make numerical execution admissible.
This core product API is not runtime/native authored-family dispatch.

Authenticated acoustic results can publish native pressure and particle velocity into
the existing recovered-field artifact and canonical partition contracts. Pressure is a
complex nodal scalar; velocity is a complex quadrature-volume average per element,
not a nodal or integration-point value. Shared entity domains are `Nodes`,
`VolumeElements` and `VolumeIntegrationPoints`; specialized structural domains remain.
`FieldTopologyIdentity::VolumeMesh` carries canonical mesh storage authority. Structural
recovery keeps its explicit family authority, including explicit/mixed topology; a
same-digest wrong variant may not substitute for either admission. Old domain/count
spellings have no compatibility decoder.

The typed acoustic state retains the `exp(+i omega t)` convention, positive frequency,
ordered node/element IDs, exact producer/source, recovery request and shared native
artifact. Publication copies already recovered values; it does not perform another
solve. Cold validation checks finite host storage, complete typed definitions, exact
native recovery plan, frequency samples and source/topology binding. The authored
tetrahedron fixture exercises compilation through cold state encoding for Tet4/Tet10;
its prescribed-pressure/residual/power checks are integration evidence, not an
independent arbitrary-tetrahedron continuum reference.

The separate matched-duct field tests compare published pressure with
`exp(-i*k*x)` and volume-average velocity with an independent simplex-moment
exponential integral divided by `rho*c=6`. For the straight tetrahedron with corner
coordinates `x_j`, the exact polynomial average is
`mean(x^n)=6*h_n(x_0,x_1,x_2,x_3)/((n+1)(n+2)(n+3))`, where `h_n` is the complete
homogeneous symmetric polynomial. Expanding the exponential against these moments
avoids using FE recovery quadrature as its own reference. Partition reconstruction,
foreign result/mesh/plan, same-digest wrong topology, node-order and shape rejection
are separate publication checks. These bounded coarse field checks do not replace
the refinement/phase/power qualification above.

The acoustic-filter suite passes 33/33, including the independent references and
authored/cold-state/detached-profile checks above. Shared field/observation regressions
previously passed 123/123; current resource regressions pass 124/124, the execution
bridge passes 91/91, and native acoustic execution/publication passes 9/9. Architecture checking still
reports five unrelated runtime module-size violations. This is not unrestricted suite
or qualification evidence. Numerical receipts,
output-policy-neutral entry intent, whole-study worker coordination, shared observation
catalog admission, independent frequency fan-out/fan-in and durable accepted-entry
sweep restart remain required. Storage topology is distinct from interpolation
projection; acoustic fields are not forced through the legacy projection-specific
catalog constructor or assigned fictitious integration-point counts. No stage,
process-restart, measured-memory or distributed-algebra gate is promoted.

### Canonical native task and resource admission

`prepare_acoustic_harmonic_task` prepares an explicit CPU task with exactly a canonical
acoustic workload, sealed mesh and typed resource plan. It is not authored-language family
dispatch. Its numerical identity binds the workload and topology, independent of transport
chunking and placement. The native worker rederives the exact resource plan and checks the
authored step, numerical/execution identities and output contract before invoking the canonical
solver. It emits a source/mesh-bound canonical result and a source-bound shared
`StatePublicationProfiles` catalog through the shared artifact bridge. Result import
authenticates both roots, their exact closed transport inventory and the rederived catalog.

The existing shared `PublishState` task consumes that result, catalog, exact publication
request and rederived resource plan. It has no mesh or acoustic-workload inputs and does
not solve again. Borrowed profiles interleave the already completed pressure/velocity arrays
without constructing recovered buffers or a semantic state. The result's exact persisted
single-frequency declaration is the authority; an ambiguous low-level sweep has an empty
publication inventory rather than an invented single-frequency state. Multi-frequency authored
dispatch runs through the governed frequency sweep and publishes one sample per authored
frequency; durable sweep continuation remains a separate gate.

The assembly-free planner counts canonical local operator supports, full real-block Krylov
workspace, shared property/projection metadata, source evidence, input materialization and
output construction/transport, including catalog transport and retained profile metadata.
Resource estimation revision40 includes both outputs. Actual source identifiers,
publication declaration strings and formulation metadata are profiled;
output fields use conservative typed serialization ceilings. The shared serial executor checks
actual prepared outputs against their admitted envelope. Content-addressed transport is counted
by unique chunks, separately from logical payload and decoded residency.

Tests cover driven free-DOF Tet4/Tet10 native execution, chunk-independent result identity,
serial artifact round trips, exact resource-planned scheduler submission, and rejection of valid
but substituted resource plans. Numerical equality with a hot solve is execution evidence, not
an independent physics reference; duct/cavity convergence and power references above remain
the independent numerical evidence. Serial artifact round trips are not process restart.

The detached publication integration reopens a filesystem object store before numerical
result import and again before shared state import, for Tet4/Tet10 and 1024/8192-byte
chunks. State identity is chunk-independent; exact producer/source, ordered IDs and native
field layout survive. Published Tet4 pressure also agrees with the hand-integrated complex
row below. Foreign numerical sources and understated catalogs reject at bridge admission;
existing shared thermal-worker tests separately retain rejection of self-consistent forged
resource/profile inputs. These are cold durable-object checks, not actual process replacement.

A separate native Tet4 check uses hand-integrated unit-right-tetrahedron coefficients:
`K44=1/18`, `M44=1/720`, `M4j=1/1440`, and `B44=sqrt(3)/72` for
`rho=3`, `c=2`, `Z=6`. Three nodes have prescribed unit pressure; the fourth complex
pressure is obtained by scalar complex division, without RunMat assembly. Exact shape-function
gradients and moments `<z>=1/4`, `<z²>=1/10` independently check recovered velocity,
intensity and energy. This is exact-element/native evidence, not continuum convergence.

Reservations are conservative requested-heap estimates, not measured process peaks or arbitrary
scale qualification. Scheduler-submission validation is not an observed remote scheduler attempt.
Whole-study lifecycle coordination, observations, independent accuracy reports, authored/generated
spectral equivalence and blocking release evidence remain required before promotion.

```sh
cargo test --locked -p runmat-analysis-fea --lib physics::acoustic::canonical
cargo test --locked -p runmat-analysis-fea --lib planning::resource::acoustic
cargo test --locked -p runmat-execution-runner-native --lib analysis_kernel::acoustic_harmonic
cargo test --locked -p runmat-analysis-fea --lib
cargo clippy --locked -p runmat-analysis-fea --all-targets --no-deps -- -D warnings
```

The executable cases also reject missing faces, overlapping explicit boundaries,
zero frequency and material damping. Radiation rejects interior faces and missing,
multiple or invalid owning media. Field recovery returns
nodal complex pressure and volume-averaged element velocity, with immutable mesh
projection identity, source evidence and power metrics.

Stage 8 remains open. Required follow-on work includes wider Tet10/curved continuum envelopes,
heterogeneous interfaces beyond the [aligned planar two-fluid benchmark](acoustic-material-interface.md),
cavity eigenfrequencies and resonance/uniqueness diagnostics,
wider exterior-radiation references, complex/frequency-dependent impedance where advertised,
rotated/nonmatching periodic/Bloch constraints, incident-field support, wider explicit/generated spectrum
equivalence, measured resource qualification and accuracy reports, broader authored native-study qualification,
observations/storage and cold restart, parallel frequency execution and blocking release
evidence. These library tests do not qualify those gates or remove the graph route's
remaining caller migration obligation.
