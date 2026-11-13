# FEA-RELEASE-CI-001: Blocking attested evidence in the existing release graph

0 PASS, 1 PARTIAL, 0 FAIL, 5 OPEN, 0 TAGGED. Generated; do not edit.

## preserved_existing_release_graph

Evidence kind: PublicationIntegrity. Criterion: Treat the current `origin/dev` CI and release job set as the baseline. Preserve its job IDs, display names, matrices, dependencies, action versions, and existing coverage; this feature branch must not cause job renames or parallel branch-specific workflow topology. Compose FEA and meshing checks into the appropriate existing jobs. Add a job only when it has a distinct runner, resource, evidence-at...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | PARTIAL | `runmat-analysis-qualification[test:release_checks]::meshing_release_evidence_is_a_release_dependency` (crates/runmat-analysis/qualification/tests/release_checks.rs:111) - The meshing release-evidence validator passes: release promotion depends on meshing release evidence; FEA release evidence and the rest of the graph are not che... |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## complete_cross_family_corpus

Evidence kind: PublicationIntegrity. Criterion: Run the complete governed family corpus. Run all coupling dependency and interface cases. Run the complete cross-family constraint, gauge, reaction, and connector corpus. Run the complete signal, excitation, callable-materialization, SRS/response-spectrum, identity, effect-policy, and spatial-realization corpus. Run the complete semantic sequence, state-transition, activation, branch, continuation...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## attested_release_and_publication_rejection

Evidence kind: PublicationIntegrity. Criterion: No full field is gathered merely to answer a distributable observation, and no visualization operation chooses result recovery semantics. No result is published with a failed mandatory accuracy criterion, or with an indeterminate one outside an explicit markIndeterminate policy. No advertised operation silently selects a reduced model. No caller-authored flag can bypass the formulation registry, a...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

