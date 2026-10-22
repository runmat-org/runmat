# FEA-AUTHORING-OUTPUT-002: Typed source queries, recovery and retained result objects

0 PASS, 0 FAIL, 13 OPEN, 0 TAGGED. Generated; do not edit.

## typed_output_queries

Evidence kind: SemanticRealization. Criterion: Outputs are typed named queries over analysis results: The field path names its recovery location explicitly: `field.node`, `field.element`, `field.integration_point`, `field.constraint`, and the other closed location kinds are distinct typed namespaces. This is required even when one location looks conventional, because quantities such as reaction force and von Mises stress may be available at mo...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## shared_query_catalog

Evidence kind: SemanticRealization. Criterion: `max` is intentionally the ordinary RunMat operation name, not a parallel FEA-only `maximum` builtin. RunMat has one canonical operation catalog with one public semantic identity and documentation entry per operation. The catalog may contain several strongly typed overloads and lowering bindings: - `max(numeric_tensor, ...)` lowers to the ordinary RunMat array/runtime implementation; - `max(scalar...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## attributed_extrema

Evidence kind: SemanticRealization. Criterion: An extremum is not merely a number. It retains `value`, canonical location or independent-domain coordinate, owning region/entity or case, tie policy, source field/sample, and reduction evidence. For example, `output.peak_stress.value` has type `stress`, while `output.peak_stress.location` is a topology-compatible location. Vector fields require an explicit projection such as `magnitude`, `compone...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## overload_contract

Evidence kind: SemanticRealization. Criterion: The shared catalog owns, for every overload: - stable operation and overload identity; - source name, aliases, availability by frontend, and documentation category; - typed parameters, named arguments, defaults, return type, units/dimension transform, and shape; - permitted quantity domains, scalar/complex/statistical forms, topology locations, frames, and coordinates; - purity, determinism, diffe...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## shared_catalog_lowering

Evidence kind: SemanticRealization. Criterion: The `.fea` compiler and RunMat compiler query that same catalog, but choose different overloads from their input types and frontend context. Runtime bindings remain runtime-owned and FEA observation execution remains FEA-owned; sharing a semantic catalog does not force both domains through the same implementation. New FEA-specific operations such as `resultant` or a governed surface `moment` still...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## closed_query_surface

Evidence kind: SemanticRealization. Criterion: The initial output-expression surface is closed and documented: - projections and representations: `component`, `magnitude`, `real`, `imag`, `phase`, `principal`, and `von_mises` where dimensionally and physically defined; - location transfer: `recover`, with a closed output location and field-transfer method whose executable family implementation, provenance, and topology-resolved cost must exist...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## physical_query_dimensions

Evidence kind: SemanticRealization. Criterion: `principal` accepts only a real symmetric stress or strain field and requires `maximum` or `minimum`; it is not a string-dispatched tensor expression. An integral multiplies the source dimension by the selected edge, face, or volume measure. A surface stress or pressure resultant has force dimensions, while `moment` adds exactly one length dimension. Discrete reaction forces and moments already ca...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## sampling_and_interpolation

Evidence kind: SemanticRealization. Criterion: Spatial selection, interpolation, and independent-domain sampling are separate operations. The first keeps a field over an exact typed region, the second evaluates shape functions at physical positions, and the third creates a history on an explicit domain: The domain is not a label: `time` requires time coordinates, `frequency` requires frequency coordinates, `mode` requires positive integer mode...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## chunked_output_retention

Evidence kind: SemanticRealization. Criterion: Field and history outputs are lazy manifest-backed objects. Reading or plotting them requests bounded chunks. Global reductions execute partition-locally and merge deterministically. A sweep collecting scalar outputs produces a typed table/tensor keyed by its axes. Retaining every large case field must be explicit: Intermediate procedure state is published only when explicitly retained or required...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

