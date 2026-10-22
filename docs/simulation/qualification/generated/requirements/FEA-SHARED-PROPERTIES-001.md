# FEA-SHARED-PROPERTIES-001: Element-complete typed constitutive ownership

0 PASS, 1 FAIL, 79 OPEN, 0 TAGGED. Generated; do not edit.

## distinct_property_roles

Evidence kind: SemanticRealization. Criterion: Material declarations, material-domain assignments, section and layup definitions, property assignments, orientations, and boundary/interface laws are distinct authored concepts. They must not be collapsed into one generic property bag or inferred from declaration count. Meshing owns topology and persistent provenance; it does not own constitutive physics. Analysis core owns the portable typed ass...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## material_bearing_roles

Evidence kind: SemanticRealization. Criterion: The material target is a closed sum type with four formulation roles: `beam` selects material-bearing one-dimensional beam domains represented by admitted edges or explicit beam-element regions; `shell` selects material-bearing two-dimensional shell domains represented by admitted faces or explicit shell-element regions; `solid` selects three-dimensional continuum-solid domains represented by exac...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## exact_assignment_targets

Evidence kind: SemanticRealization. Criterion: The four roles are therefore formulation roles rather than synonyms for raw CAD dimensions. They are sufficient for the planned production families when combined with dedicated boundary, interface, connector, and point-element contracts. New formulations may add a new closed role only when its constitutive ownership and coverage rules differ materially; they must not introduce a generic `face`, `e...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## unique_constitutive_coverage

Evidence kind: SemanticRealization. Criterion: Every active material-bearing element must resolve to exactly one compatible material assignment or one composite constitutive owner before operator assembly. Homogeneous beam, shell, solid, and cell domains use one material assignment. A laminate or other explicitly supported composite definition owns the ordered constituent materials for the elements it covers and therefore replaces, rather than...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## optional_law_capabilities

Evidence kind: SemanticRealization. Criterion: Material declarations are capability aggregates, not bags of mandatory defaults. Mechanical, thermal, acoustic, electrical, magnetic, fluid, damping, plastic, and anisotropic laws are optional typed capabilities. A material must declare at least one capability. Air does not acquire invented elastic and thermal laws merely so it can be parsed, and a structural material does not silently acquire con...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## compiled_assignment_identity

Evidence kind: SemanticRealization. Criterion: formulation adapters borrow typed immutable views of those entries and materialize only the coefficient blocks that their operators consume. Named groups may contain multiple compatible entities and may span part occurrences. Exact entity selectors identify one occurrence. Source-topology names, importer ordinals, authored list order, hash buckets, or worker partition order are never used as compi...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## required_sections_and_layups

Evidence kind: SemanticRealization. Criterion: Material coverage is necessary but not sufficient for beams, shells, composites, or anisotropy. These remain separate closed contracts: every beam element requires exactly one beam-section assignment carrying the supported section geometry and recovery locations; every isotropic shell requires exactly one shell-section assignment carrying thickness, reference surface, and transverse-shear treatmen...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## independent_section_coverage

Evidence kind: SemanticRealization. Criterion: Explicit beam and shell elements may carry direct required section identities. Geometry-derived beam and shell generation will compile separate typed `BeamSectionAssignment` and `ShellSectionAssignment` scopes; it must not infer sections from material assignments. A direct element section reference and a scoped section assignment cannot both control the same element. The compiler rejects gaps, ove...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | FAIL | `runmat-analysis-fea[lib]::properties::preparation::tests::explicit_cache_exact_ceiling_preserves_identity_shape_and_bytes` (crates/runmat-analysis/fea/src/properties/preparation/tests.rs:344) - Explicit beam and shell fixture topology cache preserves identities, borrowed source and exact resource ceiling; no material coverage, laminate physics or distr... |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## physical_section_and_orientation_data

Evidence kind: SemanticRealization. Criterion: Section definitions and assignments are distinct. Definitions carry physical geometry and constitutive choices; assignments carry typed coverage. Beam definitions initially support explicit general, rectangular, circular, and tubular sections with canonical area, inertia, torsion, shear, recovery, and reference-axis data. Homogeneous shell definitions carry thickness, an explicit reference surface...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## assignment_qualification_corpus

Evidence kind: SemanticRealization. Criterion: Qualification must include single- and multi-solid exact CAD, repeated part occurrences, semantic groups spanning multiple bodies, explicit and geometry-derived beams and shells, mixed-dimensional models, and finite-volume cells. Positive cases cover selected and explicit all-domain scopes, isotropic and heterogeneous materials, section coverage, laminate ordering, and orientation. Negative cases ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

