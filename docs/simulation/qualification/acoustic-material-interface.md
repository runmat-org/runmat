# Canonical two-fluid acoustic material interface

This independent CPU library benchmark covers an aligned planar jump in density
and sound speed on canonical affine Tet4/Tet10 meshes. It is not a whole authored
execution, incident-source implementation or Stage 8 gate.

The unit cube has two half-length fluid regions: rho1=3 kg/m³, c1=2 m/s and
rho2=5 kg/m³, c2=4 m/s. Their impedances are Z1=6 and Z2=20 Pa s/m. Actual sealed
element-region identities select the shared property compiler's two materials;
no material averaging or synthetic interface operator is introduced. Conforming
pressure continuity and weak normal-flux continuity come from the ordinary
canonical Galerkin assembly. The right exterior face uses explicit impedance Z2.

For peak phasors with exp(+i omega t), k_j=omega/c_j and L=0.5 m, pressure is

- Left: A[exp(-ik1 x)+R exp(-2ik1 L) exp(+ik1 x)].
- Right: A T exp(-ik1 L) exp(-ik2(x-L)).
- R=(Z2-Z1)/(Z2+Z1)=7/13; T=2Z2/(Z2+Z1)=20/13.

Pressure and normal velocity are continuous at the interface. A backward wave
contributes negative axial velocity. Net transmitted power is
|A|²(1-R²)/(2Z1)=|AT|²/(2Z2). These coefficients follow the pressure/normal-velocity
interface conditions presented in the [University of Washington BEE531 wave notes](https://staff.washington.edu/mbruce/pres/Waves.html).
The benchmark derives all phase and field values directly from these continuum
relations; no RunMat solve, assembled matrix or recovered field is the oracle.
An algebra-only test checks both interface continuity equations and power partition.

The prescribed inlet is the **total** analytic pressure at x=0, sampled through the
existing shared complex pressure signal. It is not an incoming-only radiation
source and must not be advertised as incident-field support. Side walls retain
the homogeneous rigid natural condition. Frequency is0.2 Hz. The 32-term exact
uniform-tetrahedron moment expansion integrates each traveling exponential for
the advertised volume-average velocity, not a centroid approximation. The matched
outlet and inlet powers are also integrated independently from recovered pressure
and element-mean velocity, rather than the assembled boundary quadratic form.

The numerical matrix has24 exact inventory entries: both orders, cells=[2,4,8],
both A=1+0.4i and iA, and both axial and rigid45-degree-rotated geometries. Fixed
acceptance: pressure RMS reduction>3(Tet4)/>6(Tet10), velocity RMS reduction
>1.5(Tet4)/>3(Tet10), finest pressure RMS<0.004|A| Pa and velocity RMS<0.01 m/s.
Finest signed inlet/outlet and passive boundary powers must agree with the
continuum value within0.005 W. Algebraic residual<1e-10 and normalized physical
power closure<1e-8 remain separate checks. Solver policy retains shared right
Ilu0,1e-11 requested residual, restart128 and maximum8192 iterations.

Exact standalone retained-impedance candidate verification passes both the algebraic
reference and all24 numerical cases (52.68s). Finest Tet4 pressure/mean-velocity
RMS errors are4.45817e-4 Pa/3.07891e-3 m/s; Tet10 are4.94405e-7 Pa/4.72724e-6 m/s.
Analytic net power is0.0686390532544 W; finest Tet10 recovered inlet/outlet are
-0.0686211611401/+0.0686343641576 W. Both phases and rotations pass unchanged
refinement and finest gates. Strict FEA-only lib/tests lint passes. Architecture
still fails three pre-existing runtime module debts, not a new interface finding.
The equivalent local-radiation draft matrix also passed24 cases; it does not erase
that envelope's separate four failed coarse gates. No resource-cap, physical
tolerance or mesh relaxation.
Curved/nonmatching interfaces, attenuation, more general incidence, generated
multi-frequency sources, independent resonance diagnostics, actual native/cold
execution, measured resources and release qualification remain open requirements.
