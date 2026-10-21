# Authored matched-impedance duct: pending execution gate

Status: pending execution gate. The first authored run failed at source validation
before geometry/physics/publication checks: same-line statements were not supported
by the authored parser, and the generator's `ones(size(...))` calls failed the
pure-effect admission. The failed project is preserved at
`/var/folders/g6/5twygq1501g6j7__xff80g300000gn/T/.tmpAE3qHC`.
The fixture now separates authored statements onto their supported lines and
generates the identical constant phasor with arithmetic only,
`-(1+0.4i)*(0*frequency+1)/6`. No physics reference or gate was changed.
Subsequent execution passed all four Tet4 generated/explicit coarse/fine cases,
including independent physical fields and authenticated cold access. The first
Tet10 generated case then rejected power closure: at 0.2 Hz, 1,651/4,096
iterations gave relative linear residual 9.882855603602013e-9 (below 1e-8), but
normalized power error 2.819442514670425e-8 exceeded the unchanged 1e-8 gate.
Input/loss power were 0.09666662735424957/0.09666663007970963 W. The diagnostic
project is retained at
`/var/folders/g6/5twygq1501g6j7__xff80g300000gn/T/.tmpOw8Uum`.
There is no all-order execution or promotion evidence yet. The test is
`typed_fea_geometry_cli::acoustic::duct::authored_matched_duct_refines_and_reopens_full_partial_complex_components`.
It does not qualify the separately failed local-radiation velocity refinement gate.

The unit OCCT box is exported to BREP, imported through actual authored geometry,
and meshed with Tet4 or Tet10. Persistent imported source ordinals zero and one must
cover exactly the inlet x=0 and outlet x=1 planes, each area 1 m² with authenticated
outward orientation. The owning medium is rho=3 kg/m³, c=2 m/s, zero damping.
The outlet has supported real specific impedance Z=rho*c=6 Pa s/m. Unspecified
sidewalls retain the canonical rigid natural boundary. At f=0.2 Hz the inlet's
outward normal velocity is -(1+0.4i)/6 m/s. Generated function and explicit
complex signal drive the same physical source.

Using peak phasors and exp(+i omega t), the independent solution is
p=(1+0.4i) exp(-ikx), k=omega/c, v_x=p/Z, v_y=v_z=0. Mean signed intensity is
|A|²/(2Z)=1.16/12 W/m² in +x. Positive incoming source power and outgoing
impedance power are each 1.16/12 W. References are analytic, not another discrete
solve or regenerated retained field. Element velocity is a physical volume average;
uniform tetrahedral moments integrate the exponential independently. The test
requires affine Tet10 midsides before using this formula. Its 32-term series at
|kx|<=pi/5 has remainder below 1e-40.

Eight actual native producer processes cover both orders, target sizes 0.5 and
0.25 m, and generated/explicit drives. Before execution the gates are fixed:
generated/explicit mesh digests equal and packed field differences below 1e-11;
more nodes and strictly decreasing pressure/velocity RMS errors under refinement;
finest relative nodal pressure RMS below 0.004 and absolute element-mean velocity
RMS below 0.01 m/s; finest intensity RMS below 0.005 W/m²; boundary powers positive
and within 0.005 W of the continuum reference; accepted source evidence,
residual <=1e-8 and normalized power imbalance below 1e-8. Finest bounds reuse
the independent matched-wave envelope. Two independently generated irregular
meshes establish bounded refinement, not structured-grid convergence order or
an asymptotic-rate certificate. No bound may be loosened after a failure.

After process exit a fresh filesystem store reopens pressure and velocity. Their
advertised public cold-query contract is packed HostF64: pressure [N,2], velocity
[E,3,2], real/imaginary components interleaved. Paging returns `Real` component
values, not `ComplexInterleaved` entity values; offsets count stored scalars. The
test reads every bounded page plus a partial seven-component page beginning at
offset one (crossing entity boundaries), and compares with full retained fields.
The entire manifest object inventory is digest-verified into a distinct cold
store; trusted-root import must reopen exactly one AcousticComplex state with
exact retained fields and mesh lineage, and reject a foreign trusted root.

Execution/physics/cold-query failures preserve their temporary project and print
its location. Pending verification: coordinated root CLI build/run with
`--features occt-native --test typed_fea_geometry_cli` filtered to
`authored_matched_duct_refines_and_reopens_full_partial_complex_components`, then
scoped strict lint and regression verification. This is single-frequency CPU
evidence only: no sweep-state semantics, observations, process restart, remote,
general radiation, non-normal incidence, curved elements or validation claim.

The pending correction continues shared GMRES from its reduced coordinates with
a tighter internal residual request when physical power is not accepted. Every
continuation spends the same total authored iteration budget; physical operators,
loads, mesh and acceptance bounds are unchanged. Physical power is reevaluated
after each candidate solve; field recovery occurs only after both criteria pass.
The resource estimate reserves an additional retained reduced-coordinate vector.
Focused tests pass an independent diagonal complex-division reference at unit and
small amplitude and reject exhausted budgets. This is not a substitute for the
pending actual Tet10/refinement/cold execution gate.

The actual continuation run now passes both generated and explicit coarse Tet10
cases (686 nodes/347 elements): pressure RMS 1.678163865605321e-5, velocity RMS
6.114275566769364e-5 and intensity RMS 3.28430182874366e-5, with equal drive
results and accepted power/cold closure. The fine Tet10 generated case (size
0.25) still fails Krylov convergence within the unchanged budget. Full test
duration was 646.86 s; failed project `.tmpgag3mt` and preceding comparison
project `.tmpaADYIp` remain in the platform temporary directory. The all-order
refinement gate is still failed. Canonical non-radiation regression passes
14/14, including forced cavities around resonance; separately failed radiation
velocity convergence is not covered by that result.

The discarded bounded trial used shared invertible right coordinate scaling,
with a positive stiffness-diagonal metric projected through the shared affine
recovery map. The indefinite Helmholtz operator, original physical L2 residual,
phase lifting, total iteration budget and power acceptance remain unchanged.
Physical warm starts are transformed into solver coordinates and back; no
left-scaled residual or artificial damping is accepted. Resource model revision
47 reserves sixteen block work vectors, including the right metric and
continuation temporaries; this was an analytical heap reservation, not measured RSS.
Final focused response, diagonal-map and acoustic-resource filters pass 11, 10
and 6 tests respectively; strict FEA/bridge library Clippy passes. Canonical
non-radiation regression passed 14/14 in 38.45 s. The actual refined CAD duct run
still failed at fine generated Tet10 within the unchanged iteration budget in
633.25 s. Failed project `.tmpN2aKSq` and comparison `.tmpEIGlXW` are retained.
The trial is removed: projected metric helper/tests, warm-coordinate conversion,
acoustic scaling and its extra workspace reservation. The existing committed
structural right preconditioner is unchanged. Resource model returns to revision
46 and thirteen block vectors for the preserved power-continuation correction.
The all-order execution gate remains failed; no bound, mesh or budget is relaxed.
