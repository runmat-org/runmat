# FEA-PUBLIC-RESULTS-001: Results readable, verdicted and viewable

0 PASS, 0 FAIL, 12 OPEN, 0 TAGGED. Generated; do not edit.

## scripts_run_and_read

Evidence kind: PublicRun. Criterion: A MATLAB script runs a `.fea` study by path with `fea.run`, reads fields with `fea.results` and `fea.field`, and compares the peak with beam theory, printing the three differences the runtime page shows.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## verdicts_and_reassess

Evidence kind: AccuracyAcceptance. Criterion: Each published example shows a verdict of passed, failed or indeterminate against its stated criteria with evidence in the results pane and CLI; an unconverged case returns indeterminate; `fea reassess` re-evaluates criteria without re-solving.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## closed_form_beside_field

Evidence kind: IndependentReference. Criterion: The field result is shown beside a closed-form estimate for bending, torsion and conduction cases, computed from the study's own parameters.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## saved_scene_capture

Evidence kind: PublicationIntegrity. Criterion: `fea snapshot` captures each hero study into a saved scene with the views the page shows: fields with cut planes and caps, field lines, arrows, isosurfaces, per-body transparency, AC-cycle playback, one view per frequency or mode, transient playback and streamlines, and the scene replays faithfully after save (NaN, colormaps, legends, units).

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | desktop_integration | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## cold_reopen_of_results

Evidence kind: ColdReplay. Criterion: Guide and hero results save and cold-reopen in Desktop and the graphics-only web viewer with identical fields and views.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | desktop_integration | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## repeated_eigenvalue_modes

Evidence kind: IndependentReference. Criterion: Repeated eigenvalues from symmetric sections do not degrade a modal run or block its snapshot.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## provenance_recorded

Evidence kind: PublicationIntegrity. Criterion: Every result records the requested and actual backend, precision, solver method, preconditioner and any fallback stage, and a fallback never reports the requested backend as used.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## desktop_study_surface

Evidence kind: PublicRun. Criterion: The Desktop study editor, geometry viewer, results pane and agent pane operate on the published studies in the released app, with the same checks and results as the CLI.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | desktop_integration | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## result_digest_determinism

Evidence kind: ExecutionEquivalence. Criterion: The same study yields identical result digests across runs, thread counts and worker counts, with the guarantee and its scope (same build, hardware class) stated.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

