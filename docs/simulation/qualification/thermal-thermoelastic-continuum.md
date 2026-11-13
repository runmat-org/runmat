# Solved nonuniform temperature to same-mesh thermoelasticity

This qualification is an affine Tet4/Tet10 unit-cube, constant-property, steady heat conduction
and small-strain one-way thermoelastic envelope. It is not CAD generation, arbitrary-mesh
transfer, temperature-dependent material closure, GPU execution or family qualification.
The initial six-case numerical run passed (root run 4925, 15.86 s); its results below motivated
stricter order-specific gates. The ratcheted rerun passes (43260, 15.12 s), the full thermal-state
qualification group passes 11/11 (77512, 14.21 s), and strict FEA all-target lint passes (90041).
The first fixture run omitted the heat source from its closed activation inventory; that was
corrected explicitly, without changing the reference, mandatory meshes or error bounds.

The October 1 admission regression check again passes all six continuum cases
(14.48 s), along with solved thermal-to-harmonic response and prescribed-temperature
sampling. Initial-state selection now validates only initial-condition activations;
shared sequence validation still owns load/constraint identities and lifecycles.
The authored activation inventory, solved-temperature transfer and independent
gates are unchanged. Thermal planning passes 16 tests including mixed selections
and unsupported-input rejection. Retained backend reservation tests count both
primary and bounded power-refinement solves and preserve one-byte-under rejection.
These focused checks do not establish a green full regression or qualification gate.

## Independent continuum reference

The unit cube has prescribed reference temperature at x=0 and x=1, insulated lateral faces,
constant conductivity k=1 W/(m K), and uniform volumetric heat generation q=80 W/m³.
Direct integration of -k T''=q gives

    T(x)-Tref = q x(1-x)/(2k)
    heat_flux_x(x) = -q(1-2x)/2

The total applied power and prescribed-boundary removal are each q W. No discrete thermal
matrix or RunMat-generated solution is used to derive these references.

For an isotropic solid with E=100 Pa, nu=0 and alpha=1e-5 /K, impose uy=uz=0 throughout,
ux=0 at x=0 and a traction-free axial end at x=1. The lateral restrictions are essential:
this is not unrestricted free dilation. Axial mechanical equilibrium and the linear
thermoelastic law give

    ux(x) = alpha q (x²/2-x³/3)/(2k)
    sigma_xx = 0
    sigma_yy = sigma_zz = -E alpha (T-Tref)

The shear stresses vanish. Integrated transverse reaction moments are -E alpha q/(12k),
the axial reaction moment is zero, and elastic energy is E alpha² q²/(120 k²).
These follow from integrating x(1-x) and x²(1-x)², not from the finite-element operator.

## Numerical and execution chain

`pipeline::linear_static::thermal_state::qualification::continuum` uses both Tet4 and Tet10
canonical structured meshes at mandatory cell refinements 1, 2 and 4. Heat generation is an
authored shared signal/excitation and thermal/structural constraints have explicit step activation.
The temperature is actually solved, published as a typed semantic state, canonically encoded,
cold decoded, and consumed through the existing prepared constraint/observation/resource
admission path. Analytic temperatures are never injected into the transfer.

Error measurements include interior temperature and displacement, element-mean heat flux and
stress, elastic energy, global heat balance and mechanical reaction moments. Positive tensor
quadrature evaluates computed-field error against the continuum polynomials; independent
simplex moments supply exact element averages. Tet10 must reproduce the quadratic temperature
and affine mean heat flux to numerical precision. Its cubic mechanical displacement remains a
genuine refinement target. Source identity and unchanged source bytes are asserted.

## Bounded evidence and ratcheted gates

Errors are dimensionless: volume RMS temperature is divided by maximum excess temperature;
volume RMS displacement by tip displacement; element-mean flux/stress RMS by their physical
peak scales; energy and reaction moments by the corresponding exact integrals. This is not a
claim that a recovered element mean resolves its within-element stress variation.

| Metric | Tet4 observed finest error | Tet4 maximum refinement ratio / finest gate | Tet10 observed finest error | Tet10 maximum refinement ratio / finest gate |
| --- | ---: | --- | ---: | --- |
| Temperature | 0.04564 | 0.26 / 0.05 | <5.6e-14 | numerical exactness <1e-8 at every level |
| Mean heat flux | 0.10206 | 0.51 / 0.11 | <1.53e-13 | numerical exactness <1e-8 at every level |
| Displacement | 0.04748 | 0.34 / 0.05 | 0.001006 | 0.17 / 0.002 |
| Mean stress | 0.11743 | 0.50 / 0.13 | 0.004817 | 0.45 / 0.005 |
| Elastic energy | 0.07941 | 0.35 / 0.09 | 0.0002827 | 0.10 / 0.0005 |
| Reaction moment | 0.0625 | 0.26 / 0.07 | <1.2e-14 | numerical exactness <1e-10 at every level |

Successive doubling gives observed Tet4 temperature/reaction order 2 and mean-flux order 1;
displacement orders about 1.80–1.88, mean-stress 1.25–1.77, and energy 1.67–1.98.
Tet10 displacement orders are about 2.81–2.92, mean-stress 1.32–1.72, and energy 3.61–3.80.
Mean-stress evidence is intentionally bounded by its actual supported projection, rather than
claiming the displacement's convergence order for all recovered quantities. The ratios are
strictly enforced at both refinement transitions; neither mandatory meshes nor references
are changed. The coarse Tet4 mesh has only prescribed-end thermal nodes and consequently
cannot resolve interior heating: interior RMS error explicitly detects this limitation.

These figures qualify this fixture's trend and precision, not every accuracy target.
The same computed source is cold decoded before mechanics, and no additional thermal solve
or analytic reconstruction occurs during transfer.

## Remaining boundaries

This slice does not close Stage 5. Transient transfer convergence, general field-artifact source
and boundary equivalence, arbitrary transfer maps, temperature-dependent iterations, parallel
assembly/parameter studies, full measured residency/resource authority and blocking release
evidence remain required. Existing authored native/process/durable execution evidence must be
retained separately; this in-process cold decode is not an actual process-restart demonstration.
