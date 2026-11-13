# Authored incident-pressure matched duct

Status: bounded authored CPU qualification PASS; no broader family qualification.

The first native test compiled successfully, then failed authored validation before
meshing: the language checker required an empty effect set while shared callable
admission already permits deterministic input rejection (`MayThrow`). A shared
effect-set predicate now owns this same rule at both boundaries; external effects,
unknown effects and suspension remain forbidden. The unchanged matrix passes
with that contract correction. No physical limit or source was altered.

This extends the retained matched-impedance CAD duct qualification with a typed
incoming pressure wave, not a prescribed total pressure or normal velocity. The
existing exact OCCT BREP/import/meshing, native producer process, authenticated
completion closure and packed full/partial public-field checks are reused.

The inlet is the authenticated x=0 plane with outward normal -x, outlet x=1 with
normal +x, and each area is 1 m². Both own rho=3 kg/m³, c=2 m/s, zero damping;
outlet specific impedance is rho*c=6 Pa s/m and unspecified sides are rigid.
At 0.2 Hz the imposed incident peak phasor is A=1+0.4i Pa. The generated source
executes `complex(1 + 0*frequency, 0.4 + 0*frequency)` through the shared compiled
signal path; a separately authored explicit table has identical ordinates.
The imported workload must retain Pressure/AcousticIncidentPressure, generated
provenance when appropriate, and no unevaluated generator.

For exp(+i omega t), the independent matched traveling-wave solution is
p=A exp(-ikx), k=omega/c; v_x=p/(rho*c), v_y=v_z=0. Pressure references are nodal;
velocity is independently integrated over each physical affine tetrahedron using
the existing exponential moment series, not interpolated from solver pressure.
The original reference checks affine Tet10 midsides before applying this formula.
Mean intensity and net incoming/outgoing powers are |A|²/(2 rho*c)=1.16/12 W/m²
and W respectively. Incident inlet Robin bookkeeping is subtracted from both
reported power channels; it is not an extra physical outgoing loss.

Eight producer processes cover Tet4/Tet10, target sizes 0.5/0.25 m, and generated/
explicit drives. Original gates stay fixed: equal generated/explicit mesh digests;
packed field differences below 1e-11; more nodes and decreasing pressure/velocity
RMS on refinement; finest pressure RMS below 0.004, velocity RMS below 0.01 m/s,
intensity RMS below 0.005 W/m²; positive powers within 0.005 W of exact;
accepted source evidence, residual <=1e-8 and normalized power imbalance <1e-8.
The solver's existing Ilu0 policy and 4096-iteration budget remain unchanged.
Two independent irregular meshes prove bounded refinement, not asymptotic order.

All eight cases PASS in the exact isolated candidate (one matrix test,2623.21s).
Generated and explicit fields are identical for both element orders and sizes.
Tet10 coarse686 nodes/347 elements gives pressure RMS1.67724e-5,
velocity RMS6.11426e-5 and intensity RMS3.28406e-5; fine3930 nodes/2441 elements
gives2.28700e-6,1.71395e-5 and9.22482e-6 respectively. Tet4 coarse119 nodes/347
elements gives1.83712e-3,7.58622e-3 and5.75357e-4; fine574 nodes/2441 elements
gives4.85141e-4,4.46685e-3 and1.67864e-4. Both orders satisfy the original finest
physical, power, residual, projection and cold authentication requirements.
Shared purity/axis/callable/geometry tests and strict shared plus native CLI/test
lint pass. This evidence does not claim a green full release lane.

After producer exit, full fields and bounded component pages reopen from a fresh
public store. A seven-component page starting at offset one crosses entity
boundaries. The complete numerical object inventory is digest-verified into a
distinct store, imported with the trusted completion root and rejected with a
foreign root. The AcousticComplex state retains exactly the original fields and
mesh lineage; no recovery solve is performed to reopen data.

```sh
cargo test --locked -p runmat --features occt-native --test typed_fea_geometry_cli authored_incident_duct_refines_and_reopens_full_partial_complex_components -- --nocapture
```

This is a single-frequency local CPU envelope. It does not qualify authored
frequency-sample state inventories, observations, failed-process replacement,
restart, remote numerical placement, distributed Helmholtz, GPU, curved elements,
general incidence, radiation, measured resources or the full Stage 8 release gate.
The retained normal-velocity duct evidence is documented in acoustic-right-lu.md;
its older unretained pending draft is not the authority for that completed run.
