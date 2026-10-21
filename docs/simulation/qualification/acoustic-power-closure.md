# Acoustic physical-power acceptance

This correction joins algebraic residual and physical power acceptance inside the
canonical CPU Helmholtz solve. It does not qualify the full acoustic envelope or
replace spatial-error, radiation, execution/restart or release evidence.

## Authority and budget

The workload passes its explicitly authored power-balance tolerance into the
canonical solve. The tolerance must be finite and strictly between zero and one.
After a candidate linear solve, the original assembled operator supplies the
prescribed-pressure reaction flux and the original passive-boundary matrix supplies
mean loss. Natural source power uses the original complex source. The normalization
includes reactive apparent power for lossless cavities, without a dimensioned floor.

A candidate must satisfy both the original linear-residual policy and physical
power bound. Failed power acceptance requests a smaller internal residual and
continues shared restarted GMRES from physical reduced coordinates. All attempts
spend one total authored iteration budget. Exhaustion, lack of progress, nonfinite
requests and cancellation fail closed. No operator, load, mesh, damping, authored
bound or total iteration ceiling is changed. Physical field recovery occurs once,
only after both criteria accept; reporting uses that same power authority.

Resource model revision 46 reserves one additional reduced-coordinate vector:
thirteen block work vectors rather than twelve, alongside the existing Krylov
basis, Hessenberg, scalar and header reservations. Checked arithmetic rejects
overflow. This is an analytical heap envelope, not measured RSS or allocation
usage evidence.

## Verification and unresolved gates

Independent diagonal complex division checks continuation at amplitudes 1 and
1e-12. A one-iteration case rejects rather than resetting or enlarging the budget.
Independent complex dense lifting checks phase reduction at multiple amplitudes.
Canonical matched/rigid ducts, phased waves, straight/curved element recovery and
forced cavities around resonance remain separate physical references.

```sh
cargo test --locked -p runmat-analysis-fea --lib response:: -- --nocapture
cargo test --locked -p runmat-analysis-fea --lib planning::resource::acoustic
cargo test --locked -p runmat-analysis-fea --lib physics::acoustic::canonical::tests -- --skip radiation
```

Post-correction response, acoustic-resource and non-radiation canonical filters
pass 11, 6 and 14 tests respectively. Strict FEA/bridge library Clippy passes.
Actual authored coarse Tet10 generated/explicit duct cases now pass unchanged
field, power and cold-access checks where power acceptance previously failed.
The fine authored Tet10 duct still fails Krylov convergence; the local-radiation
matrix still fails four coarse Tet10 velocity-reduction ratios. Those gates are
not qualified by this correction. A subsequent right-diagonal scaling trial did
not resolve the fine case and was removed, including its extra workspace.

The complete authored refinement matrix, wider boundaries, formal spatial
accuracy, measured resources, observations, restart/placement and blocking release
evidence remain required. No Stage 8 gate is promoted.
