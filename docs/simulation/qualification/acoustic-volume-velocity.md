# Acoustic boundary volume velocity — bounded qualification

This candidate extends shared quantity/signal/excitation contracts with scalar
complex `VolumeFlowRate` (m³/s) and acoustic `VolumeVelocity`. It is total outward
flow over the complete selected exterior patch, not normal velocity (m/s), mass
flow (kg/s), or an interior monopole. Positive values are outward under the
`exp(+iωt)` convention. A negative outward flow drives a wave into the duct.

The operator uses one uniform `v_n = Q / A` over the union of unique authenticated
exterior faces. The independently requested integral is the authored `Q` once,
not a sum repeated per face or reconstructed from projection quadrature. Signed
Tri6 shape weights are preserved. Each source retains quantity, sample, action,
signal, projection identity and conservation evidence through existing contracts.
Overlapping pressure/incident/normal/volume sources and rigid/impedance boundaries
reject. Paired boundary conflicts and reduced graph execution reject explicitly.

## Independent references and current evidence

- Affine triangle moments: Tri3 integrates each shape to `A/3`; Tri6 corner
  shapes integrate to zero and edge-midpoint shapes to `A/3`. Independent cross
  products determine area, not production quadrature. Multiple-face left and
  left/right unions verify normalization once, both element orders and complex
  source-vector sign. Invalid empty/duplicate/foreign/multiple-owner/interior
  patches and non-resultant distribution reject.
- Matched uniform duct: `p = ρc Q_in/A exp(-ikx)`, `v = p/(ρc)`, and transmitted
  power `ρc |Q_in|²/(2A)`, with `ρ=3`, `c=2`, `f=0.2 Hz` and
  `Q_in=(1+0.4i)/6 m³/s` or its quadrature phase. Area1 and area2 (the latter
  rotated45 degrees), Tet4/Tet10 and meshes2/4/8 give24 cases. Pressure RMS
  convergence must exceed3× for Tet4 and6× for Tet10. Finest pressure error must
  be below`0.004 sqrt(1.16)/A`, velocity RMS below`0.01/A`, both net power channels
  within`0.005/A` of the continuum value, algebraic residual below1e-10 and
  power closure below1e-8. The existing ILU0/8192 policy is explicit.
  Velocity is compared with the exact continuum tetrahedron mean using independent
  affine-simplex factorial moments of the exponential, not the centroid value or
  production quadrature/recovery. Constant-coordinate and vertex-permutation
  limits check that oracle.
- All24 continuum cases PASS in isolated candidate. Finest Tet4 pressure errors
  are5.62185e-4 (area1) and3.75497e-4 (area2); Tet10 errors1.36820e-6 and8.55929e-7.
  Tet10 powers converge to0.0966666650 W and0.0483333325 W, versus exact1.16/12
  and1.16/24 W. Both complex phases agree. Four normal-velocity equivalence cases
  PASS, but that comparison is supplementary, not the independent reference.
  Two triangle/invalid-patch tests PASS. Command filter`total_` also ran three
  unrelated existing tests:7/7 total PASS in57.01s. Overlap test PASS1/1 includes
  both action orders for duplicate volume, normal, pressure and incident sources,
  plus rigid and impedance boundary conflicts.
  Strengthened volume filter PASS9/9 in52.42s includes the unchanged24-case matrix,
  exact-mean oracle, prescribed-pressure conflicts and periodic/Bloch conflicts
  on either paired patch for both element orders. Direct typed realization rejects
  missing resultant preservation, moment preservation, follower kinematics,
  explicit direction, wrong distribution and vector-shaped declarations.
- Full shared core PASS135/135 and execution bridge PASS168/168. Volume DSL
  signals/action PASS6/6; generated materialization PASS1/1; runtime portable
  signal/action constructors PASS2/2. Generic wrong-unit scalar input now raises
  the existing diagnostic instead of silently dropping the signal. Existing
  signal/generated/incident/doc regression PASS22/22, final extended docs and
  volume DSL PASS13/13. Final full canonical acoustics PASS37/37 in64.92s. Strict
  core/FEA/language/execution lib/tests lint PASS; runtime-wide lint still fails
  eight pre-existing findings in field_scene/render_tiles/storage. No warning
  waiver or UI edit. Shared physical introspection extraction removes the old
  signal module-size debt; only three pre-existing runtime architecture debts
  remain. TS generation and semantic generator fixtures PASS; generated contracts
  agree with the source contracts.
- Authored CAD/native/cold witness PASS for all eight cases in2362.06s: Tet4/Tet10,
  target sizes0.5/0.25 and explicit/generated complex volume-flow inputs. Each
  generated/explicit pair is identical. Finest Tet4 pressure/element-mean velocity/
  intensity errors are5.75423086e-4/4.46759829e-3/1.42830992e-4; Tet10
  2.28695658e-6/1.71394379e-5/9.22480455e-6. Original pressure, exact continuum
  element-mean velocity, intensity, power, residual and cold full/partial complex
  field limits are unchanged. Cold fields come from the original retained output,
  without another solve. Strict native CLI/test lint PASS (1m32s).

## Completion boundary

This is a bounded CPU FP64 source/duct envelope, not a Stage 8 decision.
No interior monopole, nonuniform patch, arbitrary incident wave, multi-frequency
authored publication, process restart, measured-memory, distributed or GPU claim.
Shared/runtime/DSL/native and original-field cold evidence agree. Retention uses
only exact verified source blobs; unrelated worktree changes remain outside it.
