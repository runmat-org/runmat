# FEA-RELEASE-SCOPE-001: Complete supported surface and dependency promotion

0 PASS, 0 PARTIAL, 0 FAIL, 6 OPEN, 0 TAGGED. Generated; do not edit.

## supported_surface

Evidence kind: AccuracyAcceptance. Criterion: The supported surface comprises Tet4/Tet10 linear solids, three-dimensional beams, membrane/bending shells and deliberate mixed assembly; eigenmodes and prestressed modes; modal transient/harmonic, response-spectrum correlation/missing-mass and random PSD/cross-PSD response; eigenvalue buckling and prestressed direct/modal harmonic response from compatible converged state; direct transient/harmoni...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## excluded_family_admission

Evidence kind: AccuracyAcceptance. Criterion: Turbulence (RANS, LES, DES), perfectly matched layers, boundary elements, acoustic mean flow, antennas, waveguide ports, motors, piezoelectricity, fracture, fatigue, hexahedral, wedge and pyramid elements, view-factor radiation and arbitrary material models are outside the supported surface. They are rejected with typed diagnostics and the site makes no claim about them. Advertising any of them re...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## complete_family_pipeline

Evidence kind: AccuracyAcceptance. Criterion: Each family follows authored-contract validation, frozen generated-input materialization, canonical mesh projection, typed property/constraint/source compilation, physical discretization and assembly, compatible numerical solution, physical field recovery, typed observation evaluation, accuracy evidence, governed publication and persistence. No stage has duplicate authoritative implementations or ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## qualification_dependency_order

Evidence kind: AccuracyAcceptance. Criterion: Converge the public platform baseline first and establish truthful contracts before parallel nonoverlapping work. Shared topology and algebra precede physical families. Linear structures plus semantic state transitions precede base-state linearization, buckling and prestressed modal/harmonic work. Constituent solvers meet their own gates before thermoelastic, electrothermal, conjugate heat transfe...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## compute_promotion_order

Evidence kind: AccuracyAcceptance. Criterion: Canonical topology and constraints precede canonical sparse operators/patterns, a verified serial CPU reference, resource-aware shared-memory execution, whole-study remote/independent task distribution, qualified single-accelerator sparse execution, cooperative worker-group/collective contracts, family-specific distributed algebra and justified multi-accelerator promotion. Scaling contracts arrive...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

## work_scope_isolation

Evidence kind: AccuracyAcceptance. Criterion: Implementation work stays in the analysis checkout and uses the private platform checkout only for required Desktop/Server integration. The separately owned active general RunMat checkout must not be read, modified, built, formatted, tested, committed or otherwise disturbed. Unrelated private-platform changes are never included in an FEA slice.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |

