# FEA-MESH-REALPARTS-001: Real parts mesh with a published success rate

0 PASS, 0 PARTIAL, 0 FAIL, 6 OPEN, 0 TAGGED. Generated; do not edit.

## corpus_success_rate

Evidence kind: PerformanceMeasurement. Criterion: A named corpus (bracket, heat sink, tuning fork, cantilever, blade, lugs, tower, and the website fixtures) meshes on the release build; the success rate and the wall-time and memory distributions are generated from the run and published with hardware, including failures.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## original_study_meshes

Evidence kind: PublicRun. Criterion: `runmat mesh bracket.stp` with bare defaults is accepted; the KiCad heat sink meshes and its steady study closes the ~10 W balance; the tuning fork meshes with Tet10 and returns two modes with eigen-residuals; each stays inside the authored 512 MiB and one hour envelope with whole-task resources measured.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## website_fixture_parts_mesh

Evidence kind: PublicRun. Criterion: Every STEP fixture the website studies use meshes and passes quality acceptance at the authored size: cold plate (6 x 3 mm channels), enclosure, carotid bifurcation, static mixer, journal bearing, filter, muffler, cabin, speaker hemisphere with diaphragm, room, induction gear with inductor and air, busbar, current sensor, electrode.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## twisted_and_curved_hero_parts

Evidence kind: PublicRun. Criterion: Twisted or extruded blades, lugs and towers, fused pcurve tolerances, planar curve ranges and booleans on curved STEP mesh without hangs or invalid curved elements.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## mesh_command_defaults

Evidence kind: PublicRun. Criterion: `runmat mesh` accepts defaults and writes the mesh and its evidence artifact; failures are typed and name the stage.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

## wavelength_and_feature_size_checks

Evidence kind: RejectionInvariant. Criterion: Acoustic and electromagnetic meshes are checked against wavelength or skin-depth targets at check or plan time, and a mesh coarser than the declared elements-per-wavelength rule produces a located diagnostic rather than a silent accuracy loss.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |

