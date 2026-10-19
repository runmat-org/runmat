//! Standing resource admission and scalability acceptance across FEA lifecycles.
use super::{ALL, KERNEL};
use crate::requirement::*;

pub(super) const FEA_SHARED_RESOURCES_001: Requirement = Requirement {
    id: "FEA-SHARED-RESOURCES-001",
    title: "Governed resource admission and scalable FEA lifecycles",
    stages: &[Stage::SharedAuthorities],
    acceptance: concat!(
        "Governed resource admission and scalable FEA lifecycles requires ",
        "every listed criterion in its declared supported scope. A passing component ",
        "witness does not discharge unrelated criteria or broader execution ",
        "envelopes.",
    ),
    owners: &[
        "crates/runmat-meshing/execution/src/geometry_control/allocation_phase.rs",
        "crates/runmat-meshing/execution/src/budget/allocation_phase.rs",
        "crates/runmat-meshing/execution/src/solver_terminal_kernel/source.rs",
        "crates/runmat-meshing/execution/src/solver_terminal_kernel/encoding.rs",
        "crates/runmat-meshing/curve/src/shared/codec_admission.rs",
        "crates/runmat-meshing/surface/src/surface_mesh/indexed_face/partition.rs",
        "crates/runmat-meshing/surface/src/surface_mesh/indexed_face/inventory.rs",
        "crates/runmat-meshing/surface/src/face_pipeline/streaming.rs",
        "crates/runmat-meshing/surface/src/surface_mesh/join/validation.rs",
        "crates/runmat-meshing/surface/src/surface_mesh/convergence/owned.rs",
        "crates/runmat-meshing/execution/src/surface_kernel/options.rs",
        "crates/runmat-meshing/surface/src/acceptance/types.rs",
        "crates/runmat-meshing/surface/src/face_geometry/storage.rs",
        "crates/runmat-meshing/surface/src/exact_metric/types.rs",
        "crates/runmat-meshing/surface/src/exact_metric/prepared_validation.rs",
        "crates/runmat-meshing/core/src/quality/spatial_index/packed.rs",
        "crates/runmat-meshing/surface/src/exact_cdt/recovery/planarity.rs",
        "crates/runmat-meshing/execution/src/surface_kernel/error.rs",
        "crates/runmat-meshing/surface/src/refinement/spatial.rs",
        "crates/runmat-meshing/surface/src/refinement/select/domain.rs",
        "crates/runmat-meshing/surface/src/refinement/select/vertices.rs",
        "crates/runmat-meshing/surface/src/refinement/select/batch.rs",
        "crates/runmat-meshing/surface/src/refinement/insertion/candidate/indexed.rs",
        "crates/runmat-meshing/surface/src/refinement/encroachment/influence.rs",

        "crates/runmat-cpu-execution/src/in_place_sort.rs",
        "crates/runmat-execution-runner-native/src/exact_meshing_executor/cache/maintenance.rs",
        "crates/runmat-execution-runner-native/src/exact_meshing_executor/cache/maintenance/inventory.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/admission.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_state/commit.rs",
        "crates/runmat-meshing/execution/src/solver_projection_kernel/error.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/representation.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_replay.rs",
        "crates/runmat-meshing/execution/src/subtask_cache/diagnostic_chunks.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/relocation/collapse.rs",
        "crates/runmat-canonical-codec/src/direct_decoding.rs",
        "crates/runmat-canonical-codec/src/direct_decoding/representation.rs",
        "crates/runmat-execution-runner-native/build_support/meshing_fingerprint.rs",
        "crates/runmat-execution-runner-native/build_support/meshing_dependency_sources.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/topology/queries.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/candidate/queries.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/candidate/interior.rs",
        "crates/runmat-meshing/core/src/quality/predicate/adaptive_spatial/projection.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/facet_recovery/support.rs",
        "crates/runmat-meshing/execution/src/subtask_cache/reuse.rs",
        "crates/runmat-meshing/execution/src/geometry_metric/persistent.rs",
        "crates/runmat-execution-runner-native/src/meshing_blob_cache/mod.rs",
        "crates/runmat-cli/src/commands/mesh/persistence.rs",
        "crates/runmat-geometry/core/src/model/occurrence_transforms.rs",
        "crates/runmat-meshing/curve/src/shared/geometry_metric/relations.rs",
        "crates/runmat-cpu-execution/src/lib.rs",
        "crates/runmat-cpu-execution/src/native.rs",
        "crates/runmat-meshing/execution/src/geometry_metric.rs",
        "crates/runmat-meshing/surface/src/refinement/select/protected_edges.rs",
        "crates/runmat-meshing/surface/src/exact_cdt/triangulation/build/working.rs",
        "crates/runmat-meshing/size/src/metric/index.rs",
        "crates/runmat-meshing/size/src/metric/intersection.rs",
        "crates/runmat-meshing/size/src/metric/subspace.rs",
        "crates/runmat-meshing/curve/src/shared/geometry_metric/surface_sampling.rs",
        "crates/runmat-meshing/curve/src/shared/geometry_metric/proximity.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/constraint_star.rs",
        "crates/runmat-execution-runner-native/src/machine_capacity.rs",
        "crates/runmat-analysis/fea/src/planning/resource/mod.rs",
        "crates/runmat-meshing/size/src/grading/indexed.rs",
        "crates/runmat-meshing/size/src/grading/incremental.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/working/grading.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/mutation/context.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/metrics.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/handoff.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/working/output.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/working/treatment.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/retriangulate/preparation.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/admission/embedded_nodes.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/mutation/freeze.rs",
        "crates/runmat-meshing/core/src/quality/predicate/radial.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/encroachment/clearance.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/encroachment/placement.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/insertion/cavity/ties.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/flip/diagonal.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/mutation/trial.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/mutation/trial/compaction.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/working.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/mutation/edge_changes.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/schedule.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/facet_recovery/support/topology.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/frontier.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/context.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_state.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/working.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/batch/sources.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_convergence.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/insertion/retained.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/retained_state/topology.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/retained_state/quality.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/reconnection/lookahead.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/retained_state/connectivity.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/region_seed/admission.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/insertion/location.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/insertion/mutable.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/mutable_repair/candidates.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/mutation/spatial_cells.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/insertion/topology_access/face_incidence.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/flip/crossing/inventory.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/mutable_repair/advanced.rs",
        "crates/runmat-meshing/tetrahedron/src/reconnect/boundary_links.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/duplicates.rs",
        "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/search_state.rs",
        "crates/runmat-runtime/src/analysis/storage/publication.rs",
    ],
    dependencies: &[],
    obligations: &[
        obligation(
        "post_projection_resource_admission",
        concat!(
            "Resource admission occurs again after meshing and analysis projection, when ",
            "DOF count, nonzero count, constraint structure, contact candidates, ",
            "requested observations, field sizes, and partition quality are known: Plans ",
            "estimate host and device resident bytes, pattern and value bytes, vectors, ",
            "factorization or preconditioner fill, temporary assembly buffers, ",
            "state/result/checkpoint size, observation recovery and sampling work, ",
            "reduction collectives, history volume, halo volume, wall time, artifact ",
            "bytes, egress, and relay use. Infeasible work fails or replans before ",
            "expensive assembly. All estimates and actual usage feed the existing RunMat ",
            "resource and placement evidence.",
        ),
        EvidenceKind::ResourceAdmission,
        ALL,
        ),
        obligation(
            "machine_cpu_execution",
            concat!(
                "Local CPU admission discovers usable host capacity and respects explicit ",
                "worker/thread limits. All worker slots share an aggregate CPU and memory ",
                "budget; sweep fanout and nested numerical parallelism cannot multiply it. ",
                "Meshing and numerical kernels reuse bounded persistent worker pools with ",
                "declared stack and scratch demand, efficiently sleeping while idle. ",
                "Parallel read-only evaluation and owned mutations preserve physical ",
                "acceptance, deterministic reductions, cancellation and error propagation. ",
                "Retain identical-input core-count and size-series measurements before ",
                "claiming throughput scaling; CPU reservation alone proves no speedup.",
            ),
            EvidenceKind::ResourceAdmission,
            ALL,
        ),
        obligation(
            "phase_lifetimes",
            concat!(
                "Assembly, solve, recovery, publication and cold replay declare the ",
                "original retained inventories and temporary storage that coexist in each ",
                "phase. Charge live capacities, copies, sparse support, values, point ",
                "history, codec materialization and transport buffers at their actual ",
                "ownership boundaries before allocation. A decode workspace estimate is ",
                "not a typed heap measurement or an encoding workspace estimate. Release ",
                "reservations only when their original owned inventory is released. ",
                "Failed admission leaves prior reservations and visible artifacts intact; ",
                "success cannot depend on uncharged allocations or inflated ceilings.",
            ),
            EvidenceKind::ResourceAdmission,
            ALL,
        ),
        obligation(
            "shared_immutable_inventories",
            concat!(
                "Geometry, solver mesh, sparse support, compiled physical inputs and ",
                "source artifacts are reused through governed immutable ownership when ",
                "their identities and lifetimes agree. Avoid cloning complete inventories ",
                "for each mode, field, sweep entry, accepted increment or publication ",
                "adapter. Recover requested outputs in bounded partitions and retain ",
                "complete original state required for continuation. Reuse never skips ",
                "source authentication, revision fencing, physical acceptance or complete ",
                "field bindings; hashing alone does not independently verify a result.",
            ),
            EvidenceKind::ResourceAdmission,
            ALL,
        ),
        obligation(
            "bounded_local_search",
            concat!(
                "Geometric recovery, location and local topology queries declare their ",
                "search work and retained state. Repeated local queries use maintained ",
                "incidence, local traversal or conservative spatial candidates instead ",
                "of repeatedly scanning the full inventory when locality permits it. ",
                "Candidate pruning preserves exact predicates, degeneracy handling, ",
                "deterministic ordering and topology validity. Index construction, ",
                "mutation, rollback and cycle history remain accounted and cancellable. ",
                "If a global traversal is necessary, its necessity, frequency and cost ",
                "are explicit and evidenced; no search-budget exemption is implied. ",
                "A bounded scheduling window cannot starve other unresolved sources or ",
                "be treated as exhaustive failure. Cooperative size/shape convergence ",
                "retains its accepted working state and valid indexes; unavoidable complete ",
                "handoffs have explicit triggers and measured frequency. Rejection reuse ",
                "binds the full local topology, metric and protected-boundary dependencies ",
                "or invalidates conservatively. Useful accepted progress, rejected-candidate ",
                "work and complete-state reconstructions are measured together. Synthetic ",
                "scheduling priorities verify scheduling only, not physical convergence. ",
                "Automatic geometric sizing retains the measured physical directions ",
                "of curvature, feature and gap constraints. Repeated samples cannot ",
                "invent additional density in unrelated directions. Conservative ",
                "source envelopes state their scope and approximation limits; ",
                "constructive geometric and physical acceptance remain independent.",
            ),
            EvidenceKind::ResourceAdmission,
            ALL,
        ),
        obligation(
            "scaling_evidence",
            concat!(
                "For each claimed implementation envelope, retain size-series evidence ",
                "for governing input dimensions: nodes, elements, nonzeros, boundaries, ",
                "constraints, integration points, modes, increments, sweep entries and ",
                "coupled histories as applicable. State the algorithmic work and storage ",
                "model, distinguish unavoidable mathematical costs from duplicate work, ",
                "and compare predicted demand with measured phase usage and work counts. ",
                "Evidence identifies sources, backend, measurement method and shared ",
                "inventory reuse. A small fixture, one successful study or one process ",
                "sample does not establish scalability; applicable regressions remain ",
                "open until growth and numerical equivalence are assessed.",
            ),
            EvidenceKind::ResourceAdmission,
            ALL,
        ),
        obligation(
            "governed_path_equivalence",
            concat!(
                "Local, native, durable, remote, partitioned and claimed accelerator paths ",
                "enforce the same physical identities and resource controls. Expensive ",
                "preparation, hashing, serialization, index construction, assembly and ",
                "recovery have bounded work and cancellation checkpoints, including ",
                "failure diagnostics. Cancellation and resource exhaustion preserve ",
                "transactional publication and accepted state. Cold field inspection ",
                "loads only its admitted retained data and does not rerun the solver. ",
                "Optimizations use shared domain owners rather than a guide-specific ",
                "solver, codec, scheduler or admission path.",
            ),
            EvidenceKind::ResourceAdmission,
            ALL,
        ),
    ],
    witnesses: &[
        Witness {
            id: "FEA-WITNESS-SHARED-STAGE-CACHE-WORKING-SET",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Library,
                name: "exact_meshing_executor::cache::tests::cache_lifecycle_retains_feasible_working_set_beyond_256_without_per_stage_discovery",
                source: "crates/runmat-execution-runner-native/src/exact_meshing_executor/cache/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "A feasible 1,024-receipt working set survives publication with one inventory discovery, and a cold owner preserves the set. This finite metadata fixture does not certify arbitrary model throughput, cache hit rate or whole-study memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-STAGE-CACHE-INTERRUPTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Library,
                name: "exact_meshing_executor::cache::tests::cache_lifecycle_reconciles_other_writers_and_interrupted_publications",
                source: "crates/runmat-execution-runner-native/src/exact_meshing_executor/cache/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "Serialized writers invalidate stale inventories, interrupted publication reconciles durable receipts, and unreferenced objects are reclaimed. This fixture does not independently qualify the numerical contents, transport or complete native study.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-ATOMIC-COMMIT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::admission::tests::atomic_batch::cancellation_after_provisional_point_restores_complete_topology_and_quality",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/admission/tests/atomic_batch.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "Cancellation after a real provisional point restores the complete original topology and quality under uniform and scoped metrics. It does not certify every cancellation boundary, CAD publication or whole-task quiescence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-RESOLVED-METRIC",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::admission::tests::complete_metric_trial_retains_independent_improvement_rejected_by_source_metric_proxy",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/admission/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "governed_path_equivalence", envelopes: KERNEL,
                boundary: "An actual cavity trial rejected by the source-metric proxy improves complete resolved-field quality and matches independent recomputation. This finite counterexample establishes neither arbitrary geometry convergence nor complete study acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-IN-PLACE-COOPERATIVE-ORDERING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-cpu-execution", target: CargoTarget::Library,
                name: "in_place_sort::tests::orders_both_directions_and_preserves_inventory_on_cancellation",
                source: "crates/runmat-cpu-execution/src/in_place_sort.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "In-place ordering matches standard ascending and descending order across block boundaries, and cancellation preserves the input inventory. The implementation uses no full-size scratch buffer; this finite primitive test is not an end-to-end resource or speed qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-VOLUME-ENCODING-CANCELLATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_mesh::codec::indexed::tests::public_encoding_keeps_one_shot_cancellation_distinct_from_invalid_artifacts",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/codec/indexed/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "The public prepared encoder and its profiler retain a one-shot interruption as cancellation, rather than re-querying the signal and relabeling it as invalid data. This adapter regression does not certify all interruption points or native quiescence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-VOLUME-PROJECTION-CONTROL-FAILURES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution", target: CargoTarget::Library,
                name: "solver_projection_kernel::error::tests::volume_decode_preserves_original_budget_and_failure_categories",
                source: "crates/runmat-meshing/execution/src/solver_projection_kernel/error.rs",
            },
            claims: &[CoverageClaim { obligation: "governed_path_equivalence", envelopes: KERNEL,
                boundary: "Projection retains originating deadline, allocation and cancellation failure categories through its volume decode adapter, distinguishes malformed artifacts and preserves the order-elevation stage. This finite control fixture does not establish full native admission or study acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-VOLUME-INDEXED-ROUNDTRIP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_mesh::codec::indexed::tests::indexed_volume_preserves_all_topology_provenance_and_quality_values",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/codec/indexed/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "Compact artifact transport exactly preserves topology, provenance, complete accepted histories, numeric quality and signed-zero bits under uniform/scoped metrics and identity changes. This finite kernel round-trip does not establish native study acceptance or whole-task memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-VOLUME-INDEXED-EXPANSION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_mesh::codec::indexed::tests::indexed_volume_rejects_foreign_indices_duplicate_tables_and_unadmitted_expansion",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/codec/indexed/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "Foreign indices, duplicate tables/cells and unadmitted owned expansion reject in the compact transport boundary. This finite hostile-input fixture does not inventory all validator scratch or measure RSS.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-VOLUME-DIAGNOSTIC-BINDING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_mesh::quality_replay::tests::diagnostics_reject_changed_identity_controls_topology_and_hostile_bounds",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_replay/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "governed_path_equivalence", envelopes: KERNEL,
                boundary: "Quality-stage diagnostics reject changed request/geometry/implementation identities, changed original controls, damaged geometry, truncation and byte/item ceilings. Replay remains diagnostic input and cannot manufacture complete CAD publication authority.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-NUMERICAL-CACHE-LIFETIME",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Library,
                name: "meshing_blob_cache::tests::session_retention::meshing_subtask_cache_survives_session_cleanup_without_retaining_session_objects",
                source: "crates/runmat-execution-runner-native/src/meshing_blob_cache/tests/session_retention.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "A host-lifetime bounded numerical record survives native session cleanup while temporary session objects are removed; reopened reads preserve bytes and release storage. This fixture does not promise indefinite cache retention, prevent quota eviction, authorize an accepted mesh or establish whole-study performance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-NUMERICAL-CACHE-GRANT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Library,
                name: "config::tests::meshing_cache_grant_is_resolved_before_transient_scope_and_preserves_explicit_root",
                source: "crates/runmat-execution-runner-native/src/config.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "Default numerical-cache placement is resolved before adding a transient scope, explicit host roots are preserved, and relative roots reject. Both native session constructors use this shared configuration operation. It does not independently establish payload admissibility, cross-process throughput or diagnostic replay success.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-VOLUME-DIAGNOSTIC-STORAGE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution", target: CargoTarget::Library,
                name: "subtask_cache::diagnostic_chunks::tests::multi_chunk_diagnostic_round_trip_corruption_miss_and_ceiling_release_storage",
                source: "crates/runmat-meshing/execution/src/subtask_cache/diagnostic_chunks/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "Manifest-last chunk transport reassembles exact bytes, rejects corruption and understated storage, treats missing chunks as a miss, and releases admitted storage after failure or record drop. This fixture does not certify every native store eviction schedule or whole-process memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-VOLUME-COLLAPSE-LOCAL-PLAN",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_sliver::relocation::collapse::tests::local_collapse_plan_matches_complete_reference_across_unrelated_inventory",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/relocation/collapse/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "Changed-star collapse preparation agrees exactly with the former complete-map oracle, including accepted conforming candidates, identity permutations and0/64/4096 unrelated cells. Complete candidate reconstruction and independent final validation remain global; no whole-relocation locality or native runtime claim is made.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MESH-SOURCE-LINKS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Integration("meshing_fingerprint"),
                name: "linked_source_disables_reuse_instead_of_omitting_implementation",
                source: "crates/runmat-execution-runner-native/tests/meshing_fingerprint.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "An unsupported linked source entry produces unavailable identity and cannot authorize cached work. This finite native fixture does not qualify arbitrary link graphs or a fresh numerical solve.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MESH-SOURCE-PACKAGES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Integration("meshing_fingerprint"),
                name: "nested_unrelated_package_does_not_become_a_dependency_by_directory_layout",
                source: "crates/runmat-execution-runner-native/tests/meshing_fingerprint.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "An unrelated registered Cargo package nested below a selected package is excluded by ownership, while the selected package source remains influential. This is a dependency-visibility fixture, not whole-study resume qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MESH-SOURCE-INPUT-KINDS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Integration("meshing_fingerprint"),
                name: "non_rust_package_inputs_invalidate_both_stage_fingerprints",
                source: "crates/runmat-execution-runner-native/tests/meshing_fingerprint.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "A changed JSON input in an inventoried local package invalidates both stage identities. Workspace-external generated inputs and exact active-feature reachability are not established by this fixture.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MESH-SOURCE-OUTPUTS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Integration("meshing_fingerprint"),
                name: "package_output_changes_preserve_both_stage_fingerprints",
                source: "crates/runmat-execution-runner-native/tests/meshing_fingerprint.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "Package target/artifacts/.artifacts changes preserve stage identity while src/artifacts remains source. The actual Cargo watch plan excludes the package root and watches its source subtree; this does not measure cache throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MESH-SOURCE-DIRECTORIES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Integration("meshing_fingerprint"),
                name: "nested_source_target_directory_invalidates_both_stage_fingerprints",
                source: "crates/runmat-execution-runner-native/tests/meshing_fingerprint.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "A regular source module below src/target changes both stage identities. This checks source visibility, not a production cache hit or numerical mesh accuracy.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CACHE-READ-CLEANUP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Library,
                name: "meshing_blob_cache::tests::corrupt_reads_and_cancelled_reads_release_the_whole_unpublished_inventory", source: "crates/runmat-execution-runner-native/src/meshing_blob_cache/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "Corrupt and cancelled reads release unpublished storage while preserving typed errors. No invalid cache payload becomes numerical authority.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CACHE-READ-STORAGE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Library,
                name: "meshing_blob_cache::tests::repeated_reads_release_encoded_storage_but_retained_materialization_stays_charged", source: "crates/runmat-execution-runner-native/src/meshing_blob_cache/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "Repeated native cache reads release encoded buffers and unused decode workspace. A simultaneously live entry or conservatively retained decoded result cannot escape admission. This is requested storage accounting, not measured RSS.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-ENCODING-STORAGE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution", target: CargoTarget::Library,
                name: "geometry_metric::tests::persistent::repeated_metric_encoding_releases_temporary_storage", source: "crates/runmat-meshing/execution/src/geometry_metric/tests/persistent.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "The production metric cache encoder releases its temporary workspace across repeated writes of the same original metric. This does not independently establish native cache throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-GEOMETRY-STORAGE-CLEANUP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution", target: CargoTarget::Library,
                name: "geometry_control::tests::temporary_reservations_release_on_error_and_unwind", source: "crates/runmat-meshing/execution/src/geometry_control/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "Error returns and unwinding release temporary reservations without erasing peak evidence. Other non-scoped allocations remain conservatively retained.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-GEOMETRY-CONCURRENT-STORAGE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution", target: CargoTarget::Library,
                name: "geometry_control::tests::concurrent_reservations_share_one_ceiling_without_losing_peak", source: "crates/runmat-meshing/execution/src/geometry_control/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "Simultaneous reservations share one ceiling and preserve the high water mark after both workers release storage. This bounds the reservation authority, not total process memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-GEOMETRY-TEMPORARY-STORAGE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution", target: CargoTarget::Library,
                name: "geometry_control::tests::repeated_temporary_storage_preserves_peak_and_retained_result_budget", source: "crates/runmat-meshing/execution/src/geometry_control/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "Repeated transient reservations preserve the high water mark without accumulating released storage. Retained decoded storage remains charged and overlapping excess is rejected. This verifies admission accounting, not allocator RSS or a whole-study envelope.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-NESTED-ALLOCATION-LIFETIMES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution", target: CargoTarget::Library,
                name: "geometry_control::tests::scopes::nested_scopes_release_scratch_and_transfer_only_live_results", source: "crates/runmat-meshing/execution/src/geometry_control/tests/scopes.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "Nested iteration scopes release temporary allocations, transfer only retained storage and preserve global work counts and the original shared ceiling. This is declared storage accounting, not allocator RSS or whole-model acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COLD-METRIC-OWNERSHIP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution", target: CargoTarget::Library,
                name: "geometry_metric::tests::persistent::fresh_and_cold_metric_retention_excludes_preparation_scratch", source: "crates/runmat-meshing/execution/src/geometry_metric/tests/persistent.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "Fresh and cold prepared metrics retain only their reported carrier inventory without retaining construction scratch or hostile-decoder workspace. Their vector capacities can legitimately differ. This finite component witness does not establish whole-study memory or arbitrary CAD throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PHYSICAL-QUALITY-BOUNDARY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_quality::tests::thresholds::scheduling_ties_do_not_waive_full_value_physical_bounds",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/tests/thresholds.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "Above-bound ratios inside the same scheduling bucket, including the next representable value above one, remain physical violations. Nonfinite ratios fail closed. This checks the acceptance predicate; it does not establish meshing convergence or a physical error bound for arbitrary geometry.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PHYSICAL-QUALITY-QUEUE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_quality::tests::thresholds::independent_admission_and_working_queues_preserve_near_boundary_violations",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/tests/thresholds.rs",
            },
            claims: &[CoverageClaim { obligation: "governed_path_equivalence", envelopes: KERNEL,
                boundary: "Size and shape violations inside a scheduling tie are established from primitive physical inequalities, independently admitted and retained by the mutable work queue. This finite mesh family does not qualify an original CAD model or substitute for final physical admission.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PHYSICAL-QUALITY-SUBSTITUTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_quality::tests::thresholds::near_bound_report_substitution_cannot_hide_an_independent_physical_violation",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/tests/thresholds.rs",
            },
            claims: &[CoverageClaim { obligation: "governed_path_equivalence", envelopes: KERNEL,
                boundary: "A substituted ratio and missing work source cannot hide a primitive size-bound violation, even when the substitution lies within numeric field-comparison allowance. This separates approximate evidence comparison from physical acceptance; other tampering and whole-study envelopes require their own witnesses.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COLD-CODEC-PROTOCOL",
            locator: WitnessLocator::CargoTest {
                package: "runmat-canonical-codec", target: CargoTarget::Library,
                name: "direct_decoding::tests::visitors::method_dispatch_and_discarded_payloads_match_owned_value_semantics",
                source: "crates/runmat-canonical-codec/src/direct_decoding/tests/visitors.rs",
            },
            claims: &[CoverageClaim { obligation: "governed_path_equivalence", envelopes: KERNEL,
                boundary: "Permissive typed visitors compare method restrictions, numeric dispatch, owned strings, containers and discarded option/newtype/enum payloads with the original owned JSON representation. This finite protocol family is not whole-study cold replay or arbitrary custom-Serde equivalence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COLD-CODEC-HOSTILE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-canonical-codec", target: CargoTarget::Library,
                name: "direct_decoding::tests::hostile_byte_edits_and_truncation_match_original_acceptance",
                source: "crates/runmat-canonical-codec/src/direct_decoding/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "governed_path_equivalence", envelopes: KERNEL,
                boundary: "All truncations and single-bit edits of a mixed canonical fixture compare admission and decoded values with the original complete decode/re-encode oracle. The family covers representative hostile grammar; it does not establish exhaustive hostile-input coverage or measured native memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COLD-CODEC-PREALLOCATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-canonical-codec", target: CargoTarget::Library,
                name: "direct_decoding::tests::shape_limits_apply_before_any_typed_construction",
                source: "crates/runmat-canonical-codec/src/direct_decoding/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "Byte, collection, string and depth rejection plus malformed terminal UTF-8 occur before an observable typed constructor is entered. This verifies admission ordering, not a universal typed allocation bound or a lower whole-task resource reservation.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MESH-SOURCE-CLOSURE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Integration("meshing_fingerprint"),
                name: "transitive_math_adapter_and_configuration_changes_invalidate_consumers",
                source: "crates/runmat-execution-runner-native/tests/meshing_fingerprint.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "Actual build-helper fixtures bind both stage fingerprints to transitive build dependencies, native adapters and configuration changes. Optional local dependencies remain conservatively included; this is not exact active-feature reachability or an observed production cache hit.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MESH-SOURCE-SEPARATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Integration("meshing_fingerprint"),
                name: "tetrahedron_changes_invalidate_volume_and_preserve_upstream",
                source: "crates/runmat-execution-runner-native/tests/meshing_fingerprint.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "A tetrahedron-only source edit changes volume identity while preserving upstream identity in an explicit dependency fixture. Current independent geometry/mesh admission remains required on reuse; this does not qualify a fresh original CAD execution.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-FRONTIER",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::batch::sources::tests::bounded_frontier_matches_the_full_priority_order_across_unrelated_inventory",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/batch/sources/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "A batch-sized frontier matches the complete priority-sort prefix through4096 unrelated cells, including ties, empty selection and cancellation. It bounds retained source references; preparation still reads the quality inventory and does not establish complete localized refinement throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-FAIR-WINDOWS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::batch::sources::tests::bounded_windows_visit_blocked_leaders_and_the_entire_tail_before_wrapping",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/batch/sources/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "Bounded windows visit every source before wrapping on an unchanged inventory, including a blocked leading priority band. Priorities are synthetic scheduling inputs; this does not establish cavity feasibility, physical convergence or localized whole-mesh work.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-INTERRUPTED-WINDOW",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::batch::sources::tests::interrupted_window_resumes_at_the_last_examined_source",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/batch/sources/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "A partial window resumes after its last examined source rather than skipping unexamined entries or restarting the leading source. The cursor is scheduling state, not reusable rejection evidence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COUPLED-QUALITY-STATE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_mesh::quality_convergence::tests::compliant_surface_with_long_interior_edge_and_thin_collars_converges",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_convergence/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "A genuine protected octahedral collar with simultaneous size/shape defects converges under default quality criteria for uniform/scoped metrics and identity permutations. Each case observes one shared-state preparation and one terminal handoff, preserves boundary coordinates and analytic volume, and independently validates topology/quality. This finite fixture does not certify complete relocation costs, original CAD convergence or whole-task resource demand.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COUPLED-QUALITY-REJECTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_mesh::quality_convergence::tests::cancellation_and_original_candidate_budget_fail_without_a_mesh",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_convergence/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "The coupled protected-collar fixture rejects pre-cancelled execution and an insufficient original candidate-evaluation cap without returning a mesh. This covers these failure boundaries, not every interruption point or native process quiescence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PRODUCTION-REFINEMENT-LOCAL-WORK",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::working::tests::production_local_refinement_work_is_independent_of_unrelated_mesh_inventory",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/working/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "The actual production refinement step admits identical coordinates and evaluates exactly eight cells with 0/64/4096 unrelated cells under uniform/scoped metrics. Its last scoped grading update remains 16 invalidated cells and 152 edge reads. Complete preparation and terminal independent admission are outside the local measurement; this does not bound all search operations, total memory, complete-relocation work or original-model runtime.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-PARTIAL-BATCH",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::tests::batch_scheduler::valid_partial_batch_is_admitted_at_the_original_evaluation_ceiling",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/tests/batch_scheduler.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "A controlled selection fixture admits one valid ready cavity at its original evaluation ceiling while leaving unresolved quality visible. A maximum batch size is not a minimum publication size; full physical convergence and original-model performance remain separate gates.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-CHECKPOINT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::tests::batch_scheduler::cooperative_checkpoint_retains_progress_and_unresolved_quality",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/tests/batch_scheduler.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "A cooperative handoff retains the accepted insertion and unresolved quality without labeling the intermediate checkpoint converged. This verifies owner handoff, not a completed original-model mesh.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-SHAPE-HANDOFF",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::tests::batch_scheduler::shape_blocked_checkpoint_preserves_original_state_for_the_shape_owner",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/tests/batch_scheduler.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "When shape blocks floor-preserving insertion, the checkpoint moves the unchanged original topology to the shape owner and matches independently recomputed quality. Unchanged allocation pointers are bookkeeping evidence; complete shape repair and resource qualification remain separate.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-SOURCE-LOOKUP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::topology::queries::tests::canonical_source_and_local_boundary_queries_match_exhaustive_inventories",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/topology/queries/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "Canonical source lookup matches exhaustive oriented-cell search with zero,16 and4096 unrelated cells, under128 inventory comparisons per query. Local-star boundary roles match the complete boundary, including an interior vertex. This is a query bound, not complete refinement or native throughput qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-POINT-LOOKUP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::candidate::queries::tests::prepared_candidate_locations_match_exhaustive_ieee_equality_with_bounded_queries",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/candidate/queries/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "Prepared point lookup matches exhaustive coordinate equality, including signed zero and adjacent floating-point coordinates, through4096 unrelated cells with fewer than128 query probes. Preparation and complete native refinement need separate evidence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-POINT-TRANSACTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::candidate::queries::tests::accepted_point_append_is_transactional_and_preserves_original_coordinate_lookup",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/candidate/queries/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "An interrupted multi-point index update preserves the entire original lookup; successful append matches the complete old-plus-new coordinate inventory. This checks transactional point-state reuse, not measured allocator demand or physical insertion acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REFINEMENT-CANDIDATE-FAMILY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_refinement::candidate::interior::tests::lazy_interior_family_matches_complete_reconstruction_across_orientation_and_scale",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/candidate/interior.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL,
                boundary: "Lazy selection evaluates one initial lattice point while its full sequence matches complete reconstruction exactly across three scales, six coordinate orders and reflection. Point ordering and indices are checked; original-model quality and performance remain separate gates.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PROJECTED-CONSTRUCTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "quality::predicate::adaptive_spatial::projection::tests::dyadic_projection_preserves_edge_incidence_across_scale_and_plane_order",
                source: "crates/runmat-meshing/core/src/quality/predicate/adaptive_spatial/projection/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Projected predicates preserve original dyadic edge incidence and winding through five power-of-two scales and six coordinate-plane orders. This checks predicate authority used by local recovery, not query complexity, physical mesh acceptance or native throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COORDINATE-PROJECTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "quality::predicate::adaptive_spatial::projection::tests::physical_projection_requires_coordinates_without_inventing_point_identities",
                source: "crates/runmat-meshing/core/src/quality/predicate/adaptive_spatial/projection/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Coordinate-only physical projection validates finite geometry, handles all three dominant planes and degeneracy, and agrees with the captured source triangle's constructed projection. It does not assign persistent identities or establish complete recovery correctness.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-FACET-SUPPORT-INCIDENCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::facet_recovery::support::tests::captured_rounded_facet_recovers_every_original_edge_without_a_degenerate_ear",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/facet_recovery/support/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A captured four-node facet with a rounded exact midpoint recovers both support triangles, each original boundary edge once and the interior diagonal twice. This local fixture does not discharge original-model meshing, physical quality, resource or scientific gates.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-FACE-LOCAL-REUSE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution", target: CargoTarget::Library,
                name: "subtask_cache::tests::original_faces_reuse_across_partition_layouts_with_complete_boundary_identity", source: "crates/runmat-meshing/execution/src/subtask_cache/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "Fresh partition layouts reuse the same complete original faces and match uncached canonical results. Original boundary coordinate and UV changes invalidate identity; unrelated coordinates preserve it. This component witness does not establish native cache throughput or whole-study acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-CROSS-WORKER-REUSE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution", target: CargoTarget::Library,
                name: "geometry_metric::tests::persistent::fresh_workers_reuse_original_metric_and_changed_quality_invalidates_it", source: "crates/runmat-meshing/execution/src/geometry_metric/tests/persistent.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "Fresh provider instances reuse original sizing without rederivation; changed curve quality forces fresh preparation. The shared in-memory store exercises process-independent interfaces, not filesystem persistence or complete native execution.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-CACHE-ADMISSION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution", target: CargoTarget::Library,
                name: "geometry_metric::tests::persistent::stored_metric_must_fit_its_admitted_inventory_and_embedded_identity", source: "crates/runmat-meshing/execution/src/geometry_metric/tests/persistent.rs",
            },
            claims: &[CoverageClaim { obligation: "governed_path_equivalence", envelopes: KERNEL,
                boundary: "Cold sizing rejects understated decoded inventory and a foreign embedded identity. Numerical accuracy and whole-task storage require their own evidence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BLOB-CACHE-REOPEN",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Library,
                name: "meshing_blob_cache::tests::reopened_store_preserves_original_bytes_and_rejects_identity_collision", source: "crates/runmat-execution-runner-native/src/meshing_blob_cache/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "A reopened native store preserves exact retained bytes and decoded-storage inventory, and rejects conflicting bytes under one key. It does not independently establish numerical provenance or acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BLOB-CACHE-ROLLBACK",
            locator: WitnessLocator::CargoTest {
                package: "runmat-execution-runner-native", target: CargoTarget::Library,
                name: "meshing_blob_cache::tests::cancellation_rolls_back_partial_blob_and_quota_changes", source: "crates/runmat-execution-runner-native/src/meshing_blob_cache/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "governed_path_equivalence", envelopes: KERNEL,
                boundary: "Cancellation during incremental storage rolls back partial payload and quota updates while preserving previously retained bytes. This bounds one storage transaction, not all native task cancellation latency.",
            }],
        },

        Witness {
            id: "FEA-WITNESS-SHARED-CANCELLABLE-QUALITY-CONSUMPTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_quality::working::output::tests::consumed_cells_poll_mid_transfer_and_reject_lower_inventory_limits",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/working/output/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "governed_path_equivalence", envelopes: KERNEL,
                boundary: "Consuming quality-cell transfer polls cancellation during movement and rejects a lower cell-inventory limit before output allocation. It verifies this local ownership transition; it does not establish durable publication, distributed equivalence, whole-study memory bounds or throughput scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CONSUMING-QUALITY-HANDOFF",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_quality::handoff::tests::streamed_handoff_matches_full_report_and_rejects_cell_and_summary_tampering",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/handoff/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories", envelopes: KERNEL,
                boundary: "Complete cell reconstruction is compared exactly while retaining one quality report, followed by independent alternate-arithmetic validation. The fixture rejects cell, metadata, aggregate, worst-cell and inventory tampering, resource exhaustion and cancellation. It establishes this handoff's acceptance semantics, not whole-study memory bounds or throughput scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CONSUMING-TOPOLOGY-FREEZE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::mutation::tests::consuming_freeze_releases_search_indexes_before_independent_admission",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes", envelopes: KERNEL,
                boundary: "The consumed topology owner releases its shared spatial search index before the first canonical materialization checkpoint and produces the same topology as retained independent admission. This verifies ownership lifetime at the boundary, not allocator RSS, all resource estimates or a complete study's scaling envelope.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MAINTAINED-CANONICAL-LEGALIZATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::insertion::legalization::canonical_local::tests::maintained_canonical_legalization_matches_full_rebuilds_with_focus_and_barriers",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/canonical_local/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "machine_cpu_execution", envelopes: KERNEL,
                boundary: "Maintained canonical legalization matches the complete legacy operator and flip count on one, four and 32 separated neighborhoods, with sphere-inside, sphere-outside and cospherical cases, focused or unfocused queries and protected faces. It preserves symbolic policy and independently admits final topology. This checks numerical kernel equivalence, not parallel throughput, allocator demand, arbitrary relocation locality or whole-study scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PARALLEL-CAVITY-SMOOTHING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::volume_sliver::insertion::visibility::tests::prepared_parallel_smoothing_preserves_complete_serial_trajectory",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/insertion/visibility/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "machine_cpu_execution", envelopes: KERNEL,
                boundary: "Four starting points with isotropic and anisotropic metrics preserve the original serial smoothing trajectory bit for bit at one, two and four admitted CPU threads, including successful searches. This covers prepared cavity geometry and ordered trial selection, not arbitrary meshing providers, whole-study throughput, memory accounting or distributed execution.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-LOCAL-FACET-RECOVERY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron", target: CargoTarget::Library,
                name: "cdt::facet_recovery::tests::local_flip::local_flip_matches_full_operator_and_is_independent_of_unrelated_cells",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/facet_recovery/tests/local_flip.rs",
            },
            claims: &[CoverageClaim {
                obligation: "machine_cpu_execution", envelopes: KERNEL,
                boundary: "A local facet flip matches the independently rebuilt canonical operator with zero, 64 and 2,048 unrelated cells, under a fixed 128-step local search budget. This checks owned mutation equivalence and query locality; it does not establish parallel throughput, allocator demand, arbitrary cavity/refinement providers or whole-study scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PERSISTENT-CPU-POOL",
            locator: WitnessLocator::CargoTest {
                package: "runmat-cpu-execution", target: CargoTarget::Library,
                name: "native::tests::persistent_workers_ordered_results_and_explicit_budgets",
                source: "crates/runmat-cpu-execution/src/native/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "machine_cpu_execution", envelopes: KERNEL,
                boundary: "Four persistent worker identities survive repeated operations, indexed outputs match serial computation, and invalid or nested pool budgets reject. This component check does not establish native machine discovery, task admission, mesh/solve equivalence, cancellation, measured throughput or whole-study scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-INDEXED-GRADING-METADATA",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-size",
                target: CargoTarget::Library,
                name: "grading::indexed::tests::indexed_grading_handles_high_valence_without_identity_or_metadata_copies",
                source: "crates/runmat-meshing/size/src/grading/indexed/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: "A 4,097-vertex star uses non-Clone identities and preserves every original active-source allocation while indexed grading matches independent whole-graph relaxation exactly. Scratch support consists of CSR indices, scalar caps and ranked indices. This establishes neither measured allocator demand nor native study memory, cancellation, arbitrary topology mutation locality or whole-task scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-INDEXED-GRADING-EQUIVALENCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-size",
                target: CargoTarget::Library,
                name: "grading::indexed::tests::indexed_grading_matches_independent_propagation_and_preserves_metadata",
                source: "crates/runmat-meshing/size/src/grading/indexed/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories",
                envelopes: KERNEL,
                boundary: "One-, eight- and 257-vertex paths with reversed identity order, anisotropy and tied sources compare every resulting metric and evidence count with independent full-graph relaxation at grading ratios one, 1.2, two and the finite overflow boundary. This supports the shared indexed grading representation, not full physical mesh admission, native memory or arbitrary input scale.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BORROWED-TOPOLOGY-ADMISSION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::topology::validation::tests::borrowed_validation_matches_independent_reconstruction_and_rejects_tampering",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/topology/validation/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Borrowed verification matches independent complete reconstruction on an assigned and unassigned two-cell mesh. Twenty deliberately corrupted node, cell, orientation, neighbor, boundary, vertex-star and region-incidence records reject under both validators. Compact face scratch replaces a second complete mesh. This component family does not establish arbitrary assembly scaling, native peak memory or complete quality convergence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CANONICAL-TOPOLOGY-HANDOFF",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::topology::canonical::tests::construction_authority_preserves_independent_admission_and_rejects_edited_handoff",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/topology/canonical/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Construction-only immutable topology authority matches independent complete admission. Consuming the carrier discards authority; a raw handoff with broken neighbors rejects when authority is restored. This bounded component is not proof of local-only construction, arbitrary connected-mesh scaling, native peak memory or complete study acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CANONICAL-TOPOLOGY-CURRENT-CONTRACT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::topology::canonical::tests::construction_authority_does_not_waive_protected_faces_or_current_limits",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/topology/canonical/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "An immutable construction carrier still checks the current protected-face inventory, lower topology caps and immediate cancellation. Present faces pass; absent faces, tighter caps and an already cancelled small candidate reject. This component does not establish arbitrary feature recovery, native deadline latency or full physics qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COMPACT-FACE-INCIDENCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::topology::canonical::tests::compact_face_inventory_preserves_canonical_topology_under_input_permutations",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/topology/canonical/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Contiguous sorted face incidence preserves complete canonical node, cell, neighbor and boundary/region/star inventory across sixteen input permutations. Separated duplicate and same-side overlap inputs reject. This component establishes deterministic topology equivalence, not measured production-scale performance or accepted realistic native studies.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MUTATION-QUALITY-REUSE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_quality::mutation::tests::local_geometry_change_reuses_unrelated_cells_across_inventory_growth",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/mutation/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "One physical node change recomputes its incident cell while reusing all unrelated cells across one and 128-cell constant-metric inventories. Every quality record, metadata field, aggregate and selected worst cell matches complete evaluation and independent quality validation. Inventory traversal and metadata materialization remain global; this does not establish sublinear work, scoped-metric propagation, measured native memory or accepted study results.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-INSERTION-QUALITY-AUTHORITY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_quality::mutation::tests::insertion_recomputes_new_cells_and_preserves_original_quality_after_index_shifts",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/mutation/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A physical four-cell centroid insertion reuses the unchanged neighboring cell while recomputing the new cells, with inserted identities before, between and after the baseline node identities. Complete quality evidence and independent validation match fresh evaluation under unchanged constant-metric controls. This component evidence excludes general scoped-metric reuse, arbitrary mutation locality and whole native qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COMPACT-INSERTION-STATE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::retained_state::topology::insertion_tests::inserted_node_patches_restore_physical_topology_quality_and_rank_across_index_shifts",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/retained_state/topology/insertion_tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Physical one-cell insertion and compound insertion with zero or 64 unrelated cells retain only added nodes and exact changed cell records. Inserted identities precede, interleave or follow the baseline node order; materialization restores complete coordinates, orientation, neighbors, assigned regions and incidence. Independently evaluated quality and both priority spectra reconstruct exactly. Changed or removed baseline nodes, cancellation and the unchanged node ceiling reject. This component witness does not establish arbitrary mutation locality, sublinear work, whole-task native memory or quality convergence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COMPACT-QUALITY-SEED",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::retained_state::quality::tests::compact_quality_seed_preserves_every_metric_and_rank_without_unrelated_copies",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/retained_state/quality/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Independent full physical quality evaluations of a three-to-two bipyramid and zero or 64 unrelated cells reconstruct exactly from three removed and two changed quality records. Every per-cell metric, provenance field, violation ratio and aggregate extreme is preserved; both priority spectra match the original full inventories. Changed metadata on unchanged connectivity is retained, unchanged seeds contain no changed records, and noncanonical cell order rejects. This storage witness does not establish sublinear encoding work, arbitrary-mesh mutation locality, native memory or quality convergence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-OWNED-QUALITY-STATE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_refinement::tests::owned_state::admitted_state_moves_through_refinement_and_optimization_without_mesh_copies",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_refinement/tests/owned_state.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "An independently evaluated admitted tetrahedron retains the same node, cell and quality allocations through owned refinement and optimization handoffs, with no mutations and independently recomputed terminal quality. This narrow ownership witness does not measure whole-task memory or establish general quality convergence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COMPACT-RECONNECTION-SEED",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::retained_state::topology::tests::compact_seed_restores_exact_topology_and_order_with_unrelated_cells",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/retained_state/topology/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "An exact three-to-two replacement retains only three removed and two created cells against one baseline, with zero or 64 unrelated cells and distinct region assignments. Materialization restores complete physical topology, orientation, neighbors, regions and incidence; streamed ordering matches the original full candidate. Changed nodes or reassigned unchanged cells reject. No sublinear work or whole-task resource claim.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COMPACT-SEED-ADMISSION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::retained_state::topology::tests::compact_seed_materialization_preserves_cancellation_and_resource_rejection",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/retained_state/topology/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Compact reconnection seed materialization rejects cancellation and a one-cell topology ceiling for an exact two-cell candidate. This is a construction component boundary, not whole-task admission or measured native memory evidence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BOUNDED-QUALITY-FRONTIER",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::frontier::tests::streamed_retention_matches_stable_full_sort_and_bounds_live_payloads",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/frontier.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "For 4096 ranked synthetic payloads, widths zero, one, two and eight exactly match independent stable full-sort/truncate selection including ties. Retained payload counts never exceed width; construction adds at most one compact trial. Eviction/destruction release entries, and constructor failure preserves the previous beam. This carrier witness does not measure native mesh payload bytes, search cost, whole-task peak or physical quality convergence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-EXACT-LOCAL-CONNECTIVITY-KEY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::retained_state::connectivity::tests::exact_local_delta_reconstructs_connectivity_and_ignores_unrelated_cells",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/retained_state/connectivity/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "An exact three-to-two bipyramid replacement retains three removed and two created connectivity keys with zero or 64 unrelated cells. The delta reconstructs the full ordered candidate inventory, distinguishes unchanged connectivity and preserves independent full-key ordering. No sublinear comparison-work, arbitrary cavity or whole-task resource claim.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SPARSE-FACET-SUPPORT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::facet_recovery::support::topology::tests::subdivided_support_uses_local_edges_under_a_linear_work_ceiling",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/facet_recovery/support/topology/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Boundary supports with 32/128 subdivisions recover the independently specified triangle inventory under a work ceiling proportional to boundary length. Adding 64 disconnected cells leaves deterministic support unchanged. This local edge-graph family excludes a dense cubic interval inventory; it does not establish arbitrary vertex valence, connected-feature scaling, full native facet recovery, whole-task admission or measured study performance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-INCREMENTAL-INSERTION-ADMISSION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::tests::incremental_admission::admitted_insertions_discharge_each_changed_neighborhood_before_the_next_point",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/tests/incremental_admission.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Six successive insertions into a construction-verified physical prefix discharge each exactly checked neighborhood before the next point. Every prefix passes independent exhaustive empty-sphere and complete topology validation. This bounded component case does not establish arbitrary feature recovery, connected-mesh scaling, whole native admission, measured resource use or durable study results.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-INDEXED-PHYSICAL-LOCATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::location::tests::maintained_location_finds_disconnected_cells_and_exact_boundary_without_global_scan",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/location/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "After explicitly separate index construction, a bad walking hint still locates the independently specified final disconnected cell and its exact boundary, and rejects a one-step outside point across 16/256 cells. Query predicate and checkpoint bounds exclude repeated whole-inventory traversal in this family; mutation cost, arbitrary cell density, complete native study work and partial CAD loading remain separate evidence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-STRICT-ADJACENT-REDUCTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::legalization::mutable_repair::candidates::tests::adjacent_reduction_requires_strict_physical_improvement",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/mutable_repair/candidates/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "An independent radius-five construction straddles physical in-sphere equality and permits adjacent reduction only for strict improvement. Separate cospherical providers retain their own coverage/admission duties; this component is not proof of complete legalization termination, native heat recovery, arbitrary feature bounds or measured whole-task performance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-LOCAL-EDGE-NOTIFICATIONS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::trial::tests::changes::changed_edge_candidates_are_local_and_nested_rollback_coalesces",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/trial/tests/changes.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Transaction edge-loss candidates match an independent nine-edge inventory with equal checkpoint work across one and 64 disconnected neighborhoods. Nested rejection coalesces to no candidates, inner commit remains visible to its outer trial, checkpoint rejection propagates, and outer rollback restores the original topology. Recovery queue bookkeeping, arbitrary feature density, complete native admission and measured study performance remain separate gates.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ACUTE-RAY-FAMILY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::tests::acute_family::acute_incident_rays_recover_simultaneously_across_scale_and_identity_order",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/tests/acute_family.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Six independently specified acute-ray geometries recover all simultaneous chains with original endpoints and strictly ordered full parameter spans, across three scales and two identity orders. A small feature family does not establish arbitrary assembly complexity, complete native thermal or modal acceptance, resource saturation or a distributed meshing envelope.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-EXACT-RADIAL-ORDERING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "quality::predicate::radial::tests::radial_ordering_resolves_equal_and_adjacent_radii_without_overflow",
                source: "crates/runmat-meshing/core/src/quality/predicate/radial/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Exact radial ordering distinguishes equal and adjacent binary64 radii against an independent 3-4-5 radius at scales 2^-600, 1 and 2^600. This component arithmetic witness establishes neither recovery termination nor native task work, memory or timing.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ACUTE-RADIUS-PRESERVATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::encroachment::clearance::tests::acute_protection_radius_does_not_erode_through_dyadic_rounding",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/encroachment/clearance/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "An independent acute-ray discriminator exhibits coarse child-clearance-only radial erosion and requires the guarded split to retain the reference radius, clear both child spheres and approach the reference radius to floating-point resolution, in mirrored geometry at scales 0.001, 1 and 1000. The intentionally smaller rejected-radius spacing rule, full simultaneous constraint recovery, cancellation and original native performance require separate evidence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CAVITY-TIE-CONSISTENCY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::cavity::ties::tests::cavity_ties_match_the_independent_lifted_oracle_and_legalization_policy",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/cavity/ties/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Conflict-cavity inclusion and legalization share the exact symbolic in-sphere tie policy. An independent full lifted determinant on a unit-sphere inventory exercises inclusion and exclusion without changing physical coordinates. Separate forced on-face traversal, whole-topology geometric admission, cancellation and original native recovery gates remain required; this oracle does not establish global constraint preservation or performance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-INHERITED-CORNER-SPACING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::encroachment::placement::tests::rejected_corner_radius_uses_inherited_spacing_in_both_orientations",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/encroachment/placement/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Rejected radial placement uses inherited acute-corner spacing in both endpoint orientations at physical scales 0.001/1/1000, followed by separate exact child-sphere admission. The analytic discriminator distinguishes this rule from whole-fragment bisection and orthogonal projection. Does not prove global corner protection, arbitrary feature-size bounds or complete native recovery termination.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-OFFCENTRE-CLEARANCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::encroachment::clearance::tests::acute_offcentre_rounding_requires_exact_child_sphere_clearance_across_scales",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/encroachment/clearance/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Off-centre dyadic rounding clears the original witness from both materialized child diametral spheres at 0.2/0.73/2 degrees and physical scales 0.001/1/1000. Independent analytic dot products verify the exact predicate verdicts; work remains charged to the original search ceiling. Excludes all-witness clearance, midpoint policy, complete recovery termination and native resource qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-EXACT-FACE-OWNERS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::topology_access::face_incidence::tests::exact_face_owners_match_exhaustive_inventory_through_commit_and_rollback",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/topology_access/face_incidence/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Maintained mutable face owners and immutable smallest-star queries match exhaustive cell membership, including absent faces, vertex orderings, checked cavity commit and rollback. Does not prove geometric acceptance by owner lookup alone or whole native recovery termination.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-FACE-QUERY-VALENCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::topology_access::face_incidence::tests::maintained_face_query_work_is_independent_of_endpoint_valence_and_cancellable",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/topology_access/face_incidence/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A maintained interior-face query uses three checkpoints across 4/64-cell endpoint stars, returns both original owners and rejects/retries at every query checkpoint. Excludes face-index construction/update work, arbitrary mesh admission and whole-task timing/memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CROSSING-INVENTORY-REUSE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::flip::crossing::inventory::tests::read_only_crossing_assessments_share_candidates_and_mutations_rederive_them",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/flip/crossing/inventory/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Analytic convex-bipyramid crossing, obstruction and edge-contact assessments reuse one conservative inventory without additional spatial search. Checked mutation rederives candidates; rollback restores the original assessment. No cache is retained across an accepted edit, and no full recovery or native resource claim follows.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CELL-CHANGE-COALESCENCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::spatial_cells::tests::cell_change_journal_coalesces_intermediate_churn_without_changing_query_inventory",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/spatial_cells/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Sixty-four checked cavity round trips retain only five net changes at the intermediate step and zero after each exact reversal. No intermediate cell reaches the spatial index, while final query inventory agrees with original connectivity. Requires separate cancellation, canonical-handoff and geometric admission checks; excludes arbitrary changed-inventory and whole-task memory bounds.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PLANAR-PARTNER-COMPLETENESS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::flip::diagonal::tests::indexed_planar_partners_match_every_exhaustive_pair_in_original_order",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/flip/diagonal/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Exact opposite-edge face lookup preserves every candidate and its original Cartesian order across endpoint orderings and missing partners. Existing geometric admission, barrier and atomic rejection tests remain required. This candidate-equivalence witness does not establish arbitrary valence bounds, complete recovery termination, native timing or whole-task admission.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ADVANCED-PHYSICAL-REPAIR-GROWTH",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::legalization::mutable_repair::advanced::tests::advanced_physical_repair_is_local_and_independently_admitted",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/mutable_repair/advanced/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Captured four-to-four geometry repairs one dirty physical neighborhood with equal checkpoint work across 4/64 disconnected neighborhoods, after explicit index startup. Preserves point constructions and passes independent whole-result empty-sphere validation. Does not establish arbitrary neighborhood/valence/depth bounds, cold-index locality, full native timing or whole-task admission.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ADVANCED-PHYSICAL-REPAIR-CANCELLATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::legalization::mutable_repair::advanced::tests::advanced_physical_repair_cancellation_preserves_the_original_trial",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/mutable_repair/advanced/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Cancellation at every exercised checkpoint of a captured advanced physical-repair trial restores the original canonical topology and dirty inventory; a complete retry succeeds. Does not establish remote cancellation, fenced publication, arbitrary repair paths or whole-task resource/performance acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CAVITY-TRIAL-GROWTH",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::trial::tests::cavity_trial_undo_inventory_is_local_across_disconnected_growth",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/trial/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "The same checked two-to-three cavity has identical cell, corner-key, incident-star, face, boundary and free-slot undo inventories across four and sixty-four disconnected neighborhoods. Rejection restores original storage and independently reconstructed canonical topology. This component count excludes arbitrary vertex valence, cached pending-delta length, recursion depth, whole-task memory and complete native study performance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CAVITY-TRIAL-NESTING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::trial::tests::nested_cavity_trials_restore_incidence_caches_and_physical_authority",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/trial/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Committed inner changes remain reversible by their parent; rejected inner changes permit a complete outer retry and commit. Flushed spatial roots, connectivity hints, dirty-face inventory, cells and maintained incidence agree with original storage and independent canonical reconstruction. This rollback mechanism does not independently confer physical admission or complete-study qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CAVITY-TRIAL-UNWINDING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::trial::tests::cavity_trial_rolls_back_appended_points_and_unwinding",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/trial/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Unwinding after a checked cavity and point append restores the original node inventory, identity/position indexes, coordinate scale, cells, incidence and physical-state metadata; a subsequent checked edit succeeds. This component witness does not claim arbitrary whole-owner adoption inside a trial or native process cancellation.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CAVITY-TRIAL-CANCELLATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::flip::admission::tests::cancelled_local_recovery_trials_preserve_original_at_every_checkpoint",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/flip/admission/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Every tested checkpoint of an originally admitted edge-recovery transaction returns cancellation while preserving original canonical topology and an empty physical dirty inventory; retry reaches the complete original rejection. This verifies local transaction cancellation, not worker cancellation, original heat-sink acceptance or whole-task scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REPLACEMENT-MEMBERSHIP-GROWTH",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::legalization::duplicates::tests::replacement_membership_matches_exhaustive_with_local_incidence_work",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/duplicates/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Replacement membership agrees with an independent exhaustive retained-cell scan across removed-cell and reordered or duplicate-created-cell cases. Four and sixty-four disconnected neighborhoods require exactly four checkpoints for the same two-cell query. This establishes local incidence reuse for the tested membership component, not arbitrary vertex-valence bounds, full trial ownership, complete recovery or native whole-task scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REPLACEMENT-MEMBERSHIP-CANCELLATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::legalization::duplicates::tests::replacement_membership_cancellation_preserves_retained_topology",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/duplicates/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Every checkpoint of the tested indexed membership query rejects without changing retained cells; retry returns the original complete result. This read-only component evidence does not establish worker cancellation or complete task admission.",
            }],
        },

        Witness {
            id: "FEA-WITNESS-SHARED-BOUNDARY-LINK-COMPLETENESS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "reconnect::boundary_links::tests::boundary_links_match_every_exhaustive_target_and_original_cell_order",
                source: "crates/runmat-meshing/tetrahedron/src/reconnect/boundary_links/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Every target and original cell order agrees with an independent exhaustive boundary-face definition on three-, four- and seven-cell stars, including reversed target endpoints. This proves the tested topological proposal inventory; exact orientation, embedded cavity coverage, barriers and physical admission remain independent caller checks.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BOUNDARY-LINK-GROWTH",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "reconnect::boundary_links::tests::high_valence_boundary_links_have_linear_candidate_work_and_preserve_coverage",
                source: "crates/runmat-meshing/tetrahedron/src/reconnect/boundary_links/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "On closed eight- and 128-cell stars, shared boundary incidence generates the complete lexical target inventory and original cavity coverage with exactly eight checkpoints per original cell, and three candidate cells per original cell across all targets. This bounded valence series does not qualify arbitrary nonmanifold boundaries, complete recovery or native whole-task memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BOUNDARY-LINK-CANCELLATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "reconnect::boundary_links::tests::every_boundary_link_checkpoint_rejects_without_changing_source_or_queries",
                source: "crates/runmat-meshing/tetrahedron/src/reconnect/boundary_links/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Every construction and fixed-target query checkpoint rejects without changing the original boundary or retained query inventory; retry recovers the complete original cells. This pure reconnection component check does not establish worker cancellation, numerical acceptance or complete resource admission.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PHYSICAL-REPAIR-CYCLE-DELTAS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::legalization::search_state::tests::physical_repair_cycle_deltas_match_complete_connectivity_with_cavity_bounded_work",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/search_state/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Physical restoration seeds its complete connectivity hint once. Across four and sixty-four disconnected neighborhoods, prospective two-to-three cavity updates visit only two removed and three created cells, match independently rebuilt complete connectivity, preserve retained state and recognize the return cycle. Local and global repair use the same adapter. Cycle hints are not geometry identities or admission proof; this component evidence does not qualify all provider search costs, arbitrary connected-mesh scaling or native whole-task memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PHYSICAL-REPAIR-CYCLE-CANCELLATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::legalization::search_state::tests::physical_repair_cycle_seed_and_each_delta_checkpoint_reject_without_partial_state",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/search_state/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Cancellation rejects the complete seed and each removed/created-cell checkpoint without exposing a partial cycle hint or changing the retained state. This insertion-owner component check does not establish native worker cancellation, geometry admission or complete resource accounting.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REJECTED-EDGE-ADMISSION-GROWTH",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::flip::admission::tests::rejected_physical_edge_trial_preserves_original_and_avoids_global_readmission",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/flip/admission/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Across four and sixty-four disconnected physical neighborhoods, raw reconnection creates a requested edge and exact physical repair removes it. Rejection preserves the original topology and retained admission, with bounded checkpoint growth after shared index preparation. This component series does not qualify arbitrary connected meshes, complete native recovery or whole-task memory. Unverified states and protected faces retain full admission.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-POINT-PAGE-GROWTH",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "quality::spatial_index::point::tests::pages::shared_point_append_copies_bounded_pages_and_preserves_original_inventory",
                source: "crates/runmat-meshing/core/src/quality/spatial_index/point/tests/pages.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories",
                envelopes: KERNEL,
                boundary: "Point-index trials share original immutable pages. Appending inside prepared bounds at 1024 and 65536 points preserves the original complete inventory and has bounded checkpoint growth without copying every point. This component size series does not qualify arbitrary spatial distributions, full topology trial ownership or native whole-task memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-HULL-CORRIDOR-GROWTH",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::flip::boundary_corridor::tests::candidates::corridor_candidates_retain_every_exhaustive_crossing_and_prune_distant_hulls",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/flip/boundary_corridor/tests/candidates.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Prepared segment bounds retain every independently enumerated exact crossing edge and both hull face owners across four and sixty-four disconnected neighborhoods, with bounded query checkpoint growth. This broad-phase component series does not establish general connected-mesh asymptotics, complete native heat recovery or whole-task memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PHYSICAL-ADJACENCY-GROWTH",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::legalization::mutable_repair::tests::admitted::admitted_dirty_repair_is_local_and_matches_independent_whole_mesh_validation",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/mutable_repair/tests/admitted.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A construction-verified physical prefix repairs one changed cavity across four and sixty-four disconnected neighborhoods, retains original points and passes independent exhaustive empty-sphere validation with bounded checkpoint growth. This component size series does not establish connected-mesh asymptotics, native heat recovery or whole-task memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PHYSICAL-ADJACENCY-CANCELLATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::legalization::mutable_repair::tests::admitted::cancelled_admitted_repair_preserves_original_trial_and_pending_authority",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/mutable_repair/tests/admitted.rs",
            },
            claims: &[CoverageClaim {
                obligation: "governed_path_equivalence",
                envelopes: KERNEL,
                boundary: "Cancellation before local repair preserves the original trial topology and complete pending physical-adjacency inventory. This component entry cancellation does not establish every mid-repair cancellation point or native task rollback.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CONSTRUCTED-POINT-ENCLOSURES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "quality::spatial_index::point::tests::constructed_enclosures_cross_partition_regions_and_survive_append_rejection",
                source: "crates/runmat-meshing/core/src/quality/spatial_index/point/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Conservative point enclosures remain queryable across spatial partitions and after append, rebuild or rejected append. This synthetic index regression preserves broad-phase candidates; it does not independently establish exact physical predicate admission or native study acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CELL-BOUNDS-GROWTH",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::spatial_cells::tests::prepared_cell_query_prunes_distant_inventory_with_complete_candidates",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/spatial_cells/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Prepared physical cell bounds retain exhaustive candidates across four and sixty-four disconnected cavities with bounded query checkpoint growth. This broad-phase fixture does not establish connected-mesh asymptotics, exact crossing admission, native heat recovery or whole-task memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CELL-BOUNDS-TRANSACTIONS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::spatial_cells::tests::cell_bounds_follow_atomic_cavity_trials_and_canonical_node_reordering",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/spatial_cells/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "governed_path_equivalence",
                envelopes: KERNEL,
                boundary: "A persistent cell-bounds trial preserves its original owner, resumes after rejected query synchronization, rejects invalid cavity replacement and retains complete candidates after canonical node reordering. This component transaction test does not qualify a full native task or publication path.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-LOCAL-RECOVERY-REPAIR-GROWTH",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::legalization::mutable_repair::tests::local_recovery_entry_scales_with_independent_cavities_and_preserves_points",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/mutable_repair/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "The fallback recovery repair entry retains original points and independently validates four and sixty-four disconnected cavities, with bounded checkpoint growth. This entry-point size series does not establish arbitrary connected-mesh scaling, original heat recovery or whole-task native memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ADMITTED-PLANAR-RECONNECTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::flip::diagonal::tests::admitted_planar_reconnection_preserves_all_original_points_and_empty_spheres",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/flip/diagonal/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A four-cell planar cavity reconnects through exact local replacement admission, retains original physical points and passes independent exhaustive empty-sphere validation. This component fixture does not establish arbitrary recovery, native heat acceptance, whole-task memory or asymptotic scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-HULL-CORRIDOR-ELIGIBILITY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::flip::boundary_corridor::tests::interior_endpoint_rejects_hull_corridor_before_transaction_search",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/flip/boundary_corridor/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "An interior endpoint rejects a hull-only transaction using its four-cell local star under a five-step search budget, preserving original topology. Necessary-condition pruning only; not full recovery, asymptotic scaling or whole-task memory evidence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-DIRTY-FACE-REPAIR",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::insertion::legalization::mutable_repair::tests::dirty_face_work_scales_with_independent_cavities_and_preserves_points",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/legalization/mutable_repair/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Four and sixty-four independent cavities retain original points and pass exhaustive physical validation. Measured checkpoint growth bounds this disconnected repair fixture; it does not establish native heat recovery, whole-task memory or arbitrary connected-mesh scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-LOCAL-SPHERE-RECONNECTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::tests::encroached_reconnection::encroached_cospherical_edge_recovers_without_subdivision",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/tests/encroached_reconnection.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "An exact five-point encroached segment reconnects using its original points and independently passes physical Delaunay validation. This bounded-cavity degeneracy fixture does not establish large-input recovery, asymptotic scaling, whole-task memory or original heat-sink acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-LOCAL-CAVITY-SLOT-REUSE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::tests::local_replacement_reuses_lowest_slots_with_large_inactive_inventory",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A local cavity retains lowest-slot reuse and independently rebuilds valid topology with ten thousand inactive slots. This storage-ordering component check is not a heap-allocation measurement, scaling series or whole-task resource qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-POINT-REGION-TRAVERSAL",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "quality::spatial_index::point::tests::nonrectangular_region_prunes_branches_preserves_points_and_cancellation",
                source: "crates/runmat-meshing/core/src/quality/spatial_index/point/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Disjoint spatial regions preserve exhaustive original point answers, prune occupied branches and honor cancellation at every visited checkpoint. Traversal component only; no original heat-sink acceptance or whole-task memory claim.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-DIAMETRAL-ENCLOSURE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::encroachment::broad_phase::tests::sphere_pruning_preserves_exact_boundary_and_interior_across_scales",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/encroachment/broad_phase.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Outward sphere enclosure retains sampled exact-predicate interior and boundary points across scales. Exact recovery predicates retain authority. Component enclosure evidence does not qualify a complete mesh or original heat study.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ADAPTIVE-POINT-QUERY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "quality::spatial_index::point::tests::clustered_fins_match_exhaustive_closed_boundary_queries",
                source: "crates/runmat-meshing/core/src/quality/spatial_index/point/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Twelve clustered thin-fin point inventories compare adaptive candidates with exhaustive closed-boundary queries in original inventory order. Exact meshing predicates remain outside this component; no original heat acceptance or whole-task scalability claim.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ADAPTIVE-POINT-TRANSACTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "quality::spatial_index::point::tests::append_and_growth_cancel_at_every_checkpoint_without_partial_inventory",
                source: "crates/runmat-meshing/core/src/quality/spatial_index/point/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "In-bounds append cancellation exercises every checkpoint; out-of-bounds rebuilding samples the complete checkpoint range including final commit. Failed candidates preserve original point inventory and query answers and resume to the completed answer. No complete-study or measured-memory claim.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-OWNED-COPY-LIFETIMES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-runtime",
                target: CargoTarget::Library,
                name: "analysis::storage::publication::owned_storage::tests::independent_owner_phases_remain_charged_until_each_owner_is_released",
                source: "crates/runmat-runtime/src/analysis/storage/publication/owned_storage.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: "Two independently reserved temporary owners coexist with original input under a fixed ceiling. Failed growth preserves charges; consuming each reservation releases only that owner. Component bookkeeping only, without original fork completion, measured RSS or whole-task peak qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-DIRECT-CODEC-WORKSPACE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-canonical-codec",
                target: CargoTarget::Library,
                name: "direct_encoding::tests::direct_encoding_profile_bounds_original_wire_and_numeric_workspace_scales_with_bytes",
                source: "crates/runmat-canonical-codec/src/direct_encoding/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: concat!(
                    "Numeric array fixtures bound original encoded bytes and distinguish ",
                    "direct byte-buffer workspace from JSON materialization. This is a ",
                    "requested component estimate, not measured allocator usage, RSS, ",
                    "whole-task peak or complete-study scalability qualification.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-LIVE-MAP-ENCODING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-canonical-codec",
                target: CargoTarget::Library,
                name: "direct_encoding::tests::encoding_workspace_retains_ancestor_maps_and_releases_sibling_metadata",
                source: "crates/runmat-canonical-codec/src/direct_encoding/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: concat!(
                    "Sibling, wide and nested maps exercise live sorted-map metadata ",
                    "and preserve original canonical bytes. Component demand only; ",
                    "no measured allocator, whole-task peak or complete-study claim.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ORIGINAL-PROPERTY-STORAGE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: "properties::tests::original_property_storage_charges_retained_materials_and_audit_once_per_owner",
                source: "crates/runmat-analysis/fea/src/properties/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: concat!(
                    "Original material/audit ownership is separate from shared clones, ",
                    "varies with retained display buffers and survives independent cold ",
                    "validation. Conservative typed component inventory, not RSS or ",
                    "whole-study scalability qualification.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ELECTRICAL-BUFFER-CAPACITY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-core",
                target: CargoTarget::Library,
                name: "problem::materials::storage::tests::retained_material_buffers_include_unused_electrical_capacity",
                source: "crates/runmat-analysis/core/src/problem/materials/storage/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: concat!(
                    "Cleared conductivity-response storage retains its allocated capacity ",
                    "and remains charged by the material owner. Typed component demand ",
                    "only; no allocator, complete-study peak or physics claim.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CAD-BUFFER-OWNERSHIP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-core",
                target: CargoTarget::Library,
                name: "problem::materials::storage::tests::retained_material_buffers_include_cad_collection_capacity_and_target_strings",
                source: "crates/runmat-analysis/core/src/problem/materials/storage/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: concat!(
                    "Imported CAD collection capacity and persistent-target strings are ",
                    "owned separately; clearing entries releases strings while retaining ",
                    "collection storage. Typed component demand, not measured peak memory ",
                    "or complete-study scalability qualification.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-TYPED-MANIFEST-ADMISSION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-runtime",
                target: CargoTarget::Library,
                name: "analysis::storage::publication::typed_preparation::tests::retained_manifest_projection_requires_original_reservation_and_rejects_before_materialization",
                source: "crates/runmat-runtime/src/analysis/storage/publication/typed_preparation/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: concat!(
                    "A retained numeric history separately reserves original/copy storage ",
                    "and JSON projection; an insufficient ceiling rejects before the ",
                    "materializing serialization and leaves staged output empty. No ",
                    "complete-family inventory, native study or measured peak claim.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COMPILED-PROPERTY-INVENTORIES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: "properties::tests::clones_share_original_validated_properties_without_changing_cold_wire_authority",
                source: "crates/runmat-analysis/fea/src/properties/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories",
                envelopes: KERNEL,
                boundary: concat!(
                    "Tet4/Tet10 fixtures verify that compiled property clones share original ",
                    "element and audit inventory storage, preserve wire bytes, survive original ",
                    "owner drop, and independently validate cold reconstruction. No whole ",
                    "publication admission, allocator peak or study scaling claim.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MESH-TYPED-CAPACITIES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "contracts::canonical::artifact_storage::tests::unused_nested_capacities_are_counted_without_changing_canonical_identity",
                source: "crates/runmat-meshing/core/src/contracts/canonical/artifact_storage/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: concat!(
                    "Tet4/Tet10 fixtures reserve unused node and nested provenance string ",
                    "capacity. The typed storage inventory counts that growth while canonical ",
                    "bytes remain identical. Excludes allocator bookkeeping, other owner ",
                    "copies, staging admission, measured peak and whole-study scaling.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-REBASED-POINT-INDEX",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::tests::spatial::canonical_reordering_reuses_original_point_inventory_and_matches_exhaustive_queries",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/tests/spatial.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories",
                envelopes: KERNEL,
                boundary: concat!(
                    "A small admitted cavity introduces a node whose identity changes ",
                    "canonical ordering. The original point index storage is retained and ",
                    "queries match an independent exhaustive inventory before further ",
                    "append. No original heat acceptance, size-series or whole-task peak claim.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MUTABLE-CONNECTIVITY-INDEX",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::segment_recovery::flip::search_state::tests::cavity_commit_updates_streamed_keys_and_failed_replacement_preserves_them",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/segment_recovery/flip/search_state/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: concat!(
                    "One valid octahedron cavity replacement compares the maintained ",
                    "ordered-key fingerprint with independent canonical reconstruction; ",
                    "a rejected duplicate-corner replacement preserves it. No general ",
                    "spatial-candidate, growth, cancellation or whole-mesh admission claim.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PUBLICATION-PREPARATION-RESERVATIONS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-runtime",
                target: CargoTarget::Library,
                name: "analysis::storage::publication::tests::resources::preparation_reservations_accumulate_and_failed_growth_preserves_the_allowance",
                source: "crates/runmat-runtime/src/analysis/storage/publication/tests/resources.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: concat!(
                    "A 4096-byte preparation fixture accumulates two reservations and ",
                    "rejects a third without changing the retained allowance or staging ",
                    "a run. It does not establish typed-input accounting correctness, ",
                    "measured phase peak, scalable growth or complete-study admission.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-GENERAL-RESOURCE-CONTROLS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "planning::resource::tests::general::governed_general_workspace_cold_identity",
                    "_and_original_controls_are_rederived",
                ),
                source: "crates/runmat-analysis/fea/src/planning/resource/tests/general.rs",
            },
            claims: &[CoverageClaim {
                obligation: "post_projection_resource_admission",
                envelopes: KERNEL,
                boundary: concat!(
                    "Three-row general sparse count fixture verifies cold canonical resource ",
                    "plan, additive resident storage and tolerance identity; estimates are ",
                    "requested storage, not measured RSS or whole-task admission.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-GENERAL-RESOURCE-CEILING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: concat!(
                    "planning::resource::tests::general::governed_general_demand_obeys_the_origin",
                    "al_resource_ceiling",
                ),
                source: "crates/runmat-analysis/fea/src/planning/resource/tests/general.rs",
            },
            claims: &[CoverageClaim {
                obligation: "post_projection_resource_admission",
                envelopes: KERNEL,
                boundary: concat!(
                    "General-solve demand rejects the original 100-byte accuracy ceiling; no ",
                    "physical operator or remote admission evidence.",
                ),
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SCOPED-GRADING-EDIT-ORACLE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-size",
                target: CargoTarget::Library,
                name: "grading::incremental::tests::graph_edits_match_full_reference_across_cycles_ties_increases_and_deletions",
                source: "crates/runmat-meshing/size/src/grading/incremental/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Graph edits compare exact caps with the complete indexed grading oracle across cycles, ties, raw-length increases/decreases, removal/reinsertion and ratios from one through the largest finite f64. The fixture family does not establish complete native mesh performance or original-study acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SCOPED-GRADING-CONNECTED-WORK",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-size",
                target: CargoTarget::Library,
                name: "grading::incremental::tests::changed_dependency_closure_does_not_scan_unrelated_connected_inventory",
                source: "crates/runmat-meshing/size/src/grading/incremental/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A fixed source-cap increase on connected chains of 16, 128 and4096 vertices retains at most five cap entries and reads fewer than40 neighbors while matching complete grading. Actual dependency changes can legitimately propagate across the full graph; this is no constant-work guarantee for arbitrary edits.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SCOPED-GRADING-CANCELLATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-size",
                target: CargoTarget::Library,
                name: "grading::incremental::tests::cancellation_and_invalid_policy_leave_the_graph_unmodified",
                source: "crates/runmat-meshing/size/src/grading/incremental/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Cancellation interrupts a borrowed cap-update plan without modifying its admitted graph; invalid grading policy rejects, and an empty edit performs zero neighbor reads. Caller topology admission and atomic application remain separate contracts.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SCOPED-QUALITY-ROLLBACK",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_quality::working::tests::local_quality_matches_full_reference_through_inserted_identity_reordering_and_rollback",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/working/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Uniform and scoped insertion trials compare every quality field and both ranking spectra with full canonical evaluation through new identity orderings. Rollback restores original graph-slot ownership and quality. This finite family does not cover every possible metric or original CAD model.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SCOPED-TREATMENT-ORACLE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::working::tests::search::scoped_treatment_matches_complete_canonical_oracle",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/working/tests/search.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "One scoped interior-sliver treatment compares final topology, all quality records and the mutation sequence with the complete canonical orchestration oracle. Complete deletion/retriangulation locality, native resource peaks and guide-study qualification require separate evidence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-LOCAL-SEARCH-DIFFERENTIAL",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::working::tests::search::working_searches_match_canonical_candidate_selection_and_full_quality",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/working/tests/search.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "One interior-sliver fixture compares reconnection, fixed-star relocation and insertion selection against complete canonical search, including mutation sequence, topology and every quality record. Uniform and scoped face sizing are both exercised. This does not establish arbitrary mesh equivalence, deletion/retriangulation locality, measured native demand or accepted study results.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-LOCAL-EDIT-INVENTORY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::working::tests::scaling::retained_local_edits_and_quality_undo_are_independent_of_background_mesh_size",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/working/tests/scaling.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "One centroid split retains two checked edits and bounded quality undo entries with zero, 32 and256 disconnected background cells under uniform and scoped face sizing; rollback and independent complete handoff match the original state. The counter bounds retained local records, not total preparation work, allocator bytes, connected dependency propagation or native peak memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-NESTED-TOPOLOGY-UNDO",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::trial::tests::retention::parent_undo_drops_rejected_child_history_across_distinct_cavities",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/trial/tests/retention.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Rejected nested flips across four and 64 distinct disconnected cavities leave zero parent undo records; a committed child remains reversible, with exact original topology, incidence and search authority restored. This bounds retained undo differences on these fixtures, not arbitrary connected propagation, allocator footprint or native study acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-DEFERRED-SPATIAL-UNDO",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::spatial_cells::pending::tests::checkpoints_retain_only_changed_keys_across_large_pending_inventory",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/spatial_cells/pending/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Checkpoints over 32 and 65,536 deferred spatial updates retain no copied baseline records; changes retain only per-key undo. Nested rollback and committed child changes restore the entire original deferred inventory. This bounds journal entries, not allocator bytes, global spatial-tree rebuilding or native study memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-NESTED-QUALITY-UNDO",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_quality::working::tests::retention::nested_quality_undo_shares_original_evidence_and_discards_transient_history",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/working/tests/retention.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Two hundred fifty-six temporary quality entries leave only one live parent difference. Undo and comparison originals share the same immutable evidence allocation; nested commit and rollback restore the exact complete quality report. This private journal fixture does not validate physical candidate geometry, whole-task memory or throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PERSISTENT-POINT-REPLACEMENT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "quality::spatial_index::point::tests::replacement::interior_point_replacement_copies_one_page_and_preserves_the_original_index",
                source: "crates/runmat-meshing/core/src/quality/spatial_index/point/tests/replacement.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Interior replacement in 1,024- and 65,536-point inventories copies one 32-point payload page while the original index stays intact. This excludes relocation outside retained routing bounds, complete mesh mutation, measured whole-task demand and arbitrary native model scale.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-STREAMED-UNIFORM-QUALITY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_quality::metrics::tests::streamed_uniform_contexts_preserve_every_original_cell_and_complete_quality",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/metrics/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories",
                envelopes: KERNEL,
                boundary: "Uniform metric contexts stream one cell at a time and match every original context and complete quality report. This does not measure allocator footprint, scoped graph demand or native study acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SEARCH-HISTORY-SHARING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::mutation::trace::history::tests::retained_frontiers_share_long_prefixes_and_release_rejected_branches",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/trace/history/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories",
                envelopes: KERNEL,
                boundary: "Thirty-two retained branches share exact chronological prefixes at 31 and 65,535 edits, copy bounded tail handles, retain one payload per original or new edit and release rejected branches. This verifies history ownership, not native allocator peaks, whole-study memory or physical meshing acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COMPONENT-OWNED-HANDOFF",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_mesh::components::tests::single_component_moves_storage_and_preserves_independent_encoding",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/components/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: "Single-region completion retains the original node, cell and quality allocations and matches the former rebuild's complete output and independently validated canonical encoding. This excludes whole-native-study memory peaks and multi-region aggregate allocation bounds.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COMPONENT-ASSEMBLY-EQUIVALENCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_mesh::components::tests::multiple_components_match_original_assembly_and_merge_shared_provenance",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/components/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories",
                envelopes: KERNEL,
                boundary: "Consuming multi-region assembly matches the original independent rebuild including merged shared-node provenance and complete quality. This does not establish arbitrary partition scaling, native throughput or whole-task admission.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-STREAMED-ARTIFACT-CLOSURE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "contracts::canonical::chunk::tests::streaming_closure_matches_original_content_across_fragmentation_and_rejects_forged_digest",
                source: "crates/runmat-meshing/core/src/contracts/canonical/chunk/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: "Fragmented records retain the original logical digest with borrowed payload fragments and reject forged content. This component check does not establish native IO throughput or whole-task memory admission.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ACCEPTED-STAGE-REUSE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution",
                target: CargoTarget::Library,
                name: "serial_tests::reuse::accepted_stage_reuse_rebinds_authority_and_preserves_original_observations",
                source: "crates/runmat-meshing/execution/src/serial_tests/reuse.rs",
            },
            claims: &[CoverageClaim {
                obligation: "governed_path_equivalence",
                envelopes: KERNEL,
                boundary: "An accepted surface-stage receipt rebinds session access, preserves original observations and rejects changed implementation, changed input, understated limits and corrupt storage. Reuse is not fresh numerical execution or a native scaling measurement.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-IMMUTABLE-SPARSE-STORAGE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-analysis-fea",
                target: CargoTarget::Library,
                name: "algebra::matrix::tests::immutable_clones_share_numeric_storage_and_cold_decode_revalidates",
                source: "crates/runmat-analysis/fea/src/algebra/matrix/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories",
                envelopes: KERNEL,
                boundary: "Immutable sparse clones share support and numeric values; cold decoding preserves exact wire identity and rejects malformed shape. Transformed operators, solver workspaces and native heap measurements remain separate obligations.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-VERTEX-STAR-LOCAL-INVENTORY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::retriangulate::preparation::tests::vertex_star_inventory_preserves_complete_cavity_and_outside_owners",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/retriangulate/preparation/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A vertex-star neighborhood retains five cells against a 29-cell canonical mesh and reproduces the complete cavity, original outside-face owners, regions and exact target volume. This excludes full canonical reconstruction, legalization, allocator measurements and whole-study throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-VERTEX-REMOVAL-INDEX-REMAP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::retriangulate::preparation::tests::removal_index_shift_matches_canonical_identity_remapping",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/retriangulate/preparation/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Every removal position in a canonical node inventory matches identity-map remapping for all unchanged cells, and cells referencing the removed node reject. This verifies remapping semantics without establishing independent candidate admissibility or native runtime.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-STRICT-CONTAINMENT-BROAD-PHASE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::admission::embedded_nodes::tests::indexed_containment_matches_exhaustive_predicates_at_faces_edges_and_nearby_points",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/admission/embedded_nodes/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Strict containment agrees with exhaustive adaptive predicates at faces, edges and adjacent floating-point positions across three scales and three coordinate permutations. An exact open-box broad phase is not a replacement for canonical topology, Delaunay, physical quality or native throughput admission.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ALIGNED-PLANE-PRUNING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_sliver::admission::embedded_nodes::tests::aligned_plane_inventory_is_rejected_before_adaptive_predicates",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/admission/embedded_nodes/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A 1,028-node coplanar inventory produces zero strict-interior candidates with fewer than 128 spatial-query checkpoint visits and matches exhaustive containment. This bounds the declared aligned-plane query, not arbitrary occupied geometry, index construction cost, measured allocator storage or whole-study scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CURVE-BORROWED-INCIDENCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-curve",
                target: CargoTarget::Library,
                name: "shared::geometry_metric::relations::tests::incidence_construction_is_linear_and_queries_preserve_original_use_order",
                source: "crates/runmat-meshing/curve/src/shared/geometry_metric/relations/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "The 257-owner, 32768-use incidence fixture performs exactly two owner resolutions per use, and borrowed row queries preserve original use order without further owner resolution. This is component work-count evidence, not arbitrary native geometry throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PREPARED-OCCURRENCE-LOOKUP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-geometry-core",
                target: CargoTarget::Library,
                name: "model::occurrence_transforms::tests::prepared_world_lookup_matches_original_composition_for_repeated_and_nested_instances",
                source: "crates/runmat-geometry/core/src/model/occurrence_transforms/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Prepared occurrence matrices match original ancestor composition bit for bit for repeated and nested placements. Repeated queries do not repeat preparation allocation or work charges. This covers occurrence lookup, not all geometric queries or whole-model memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CPU-COMPATIBLE-OWNERS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-cpu-execution",
                target: CargoTarget::Library,
                name: "native::tests::same_capacity_owners_overlap_without_allocating_competing_pools",
                source: "crates/runmat-cpu-execution/src/native/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "machine_cpu_execution",
                envelopes: KERNEL,
                boundary: "Two concurrent owners enter before either is released at admitted capacities one and two, and ordered kernels retain the declared capacity. This checks lease concurrency and pool ownership, not whole-machine saturation, native throughput, or measured stack residency.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CPU-CAPACITY-WAIT-CANCELLATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-cpu-execution",
                target: CargoTarget::Library,
                name: "native::tests::incompatible_capacity_wait_observes_cancellation_before_the_owner_finishes",
                source: "crates/runmat-cpu-execution/src/native/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "machine_cpu_execution",
                envelopes: KERNEL,
                boundary: "An incompatible one-thread request cancels while a two-thread owner remains active; its body never executes, and the existing pool remains usable. This is controlled acquisition evidence, not cancellation latency for arbitrary native kernels.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-GEOMETRY-METRIC-SINGLE-FLIGHT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution",
                target: CargoTarget::Library,
                name: "geometry_metric::tests::concurrent_consumers_prepare_once_preserve_authority_and_admit_each_borrow",
                source: "crates/runmat-meshing/execution/src/geometry_metric/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories",
                envelopes: KERNEL,
                boundary: "Eight concurrent consumers share one actual prepared metric and match independent derivation. Changes to requested sizing or curve/surface controls require fresh preparation. This covers one worker-scoped immutable authority, not cross-process caching or native meshing acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-GEOMETRY-METRIC-CONTROLLED-REUSE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution",
                target: CargoTarget::Library,
                name: "geometry_metric::tests::cancelled_preparation_is_not_cached_and_cache_hits_still_observe_cancellation",
                source: "crates/runmat-meshing/execution/src/geometry_metric/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories",
                envelopes: KERNEL,
                boundary: "Cancellation before and during derivation does not create a successful cached field; later preparation succeeds. Cached reuse still rejects cancellation and denied storage. This does not establish whole-task memory bounds or scientific accuracy.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CLI-CONSUMING-MESH-HANDOFF",
            locator: WitnessLocator::CargoTest {
                package: "runmat",
                target: CargoTarget::Library,
                name: "commands::mesh::persistence::tests::consuming_persistence_preserves_complete_canonical_pair_and_summary",
                source: "crates/runmat-cli/src/commands/mesh/persistence/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: "Both element orders preserve the complete canonical mesh/evidence pair and scalar summary through consuming persistence and independent cold decode. This checks handoff semantics; source ownership and separate physical-footprint measurements are required to establish peak-memory scaling.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-PROTECTED-EDGE-INVENTORY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "refinement::tests::protected_edge_inventory::selection_preserves_protection_with_a_large_reversed_segment_inventory",
                source: "crates/runmat-meshing/surface/src/refinement/tests/protected_edge_inventory.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A 16384-entry reversed protected-segment inventory preserves minimum-edge protection and charges its temporary index before use; denied allocation rejects. This component lookup/admission case does not prove arbitrary constrained triangulation, native throughput, or whole-study memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-CONNECTED-CAVITY-EQUIVALENCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "exact_cdt::triangulation::build::working::tests::connected_cavities_match_exhaustive_predicates_including_degeneracies",
                source: "crates/runmat-meshing/surface/src/exact_cdt/triangulation/build/working/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Connected cavity traversal preserves the exhaustive builder’s canonical topology on grids, reversed insertion order, transformed coordinates and cocircular inputs. This finite component comparison does not establish arbitrary CAD coverage or native throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-CONNECTED-CAVITY-WORK",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "exact_cdt::triangulation::build::working::tests::bounded_local_work_handles_a_large_planar_inventory",
                source: "crates/runmat-meshing/surface/src/exact_cdt/triangulation/build/working/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A 2025-point planar inventory compares exact canonical output against the exhaustive builder while applying a predicate ceiling that the exhaustive scan exceeds. This component work bound does not establish whole-task memory, constrained refinement convergence or native study performance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-PHYSICAL-MULTIPLICITY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-size",
                target: CargoTarget::Library,
                name: "metric::tests::topology_multiplicity_preserves_density_and_all_constraint_provenance",
                source: "crates/runmat-meshing/size/src/metric/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Up to4096 applicable scopes preserve every source count while repeated physical tensors leave density unchanged, for diagonal and non-diagonal inputs. This finite component family does not qualify arbitrary spatial metric fields or complete CAD/study cost.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-COMPACT-SCOPE-LOOKUP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-size",
                target: CargoTarget::Library,
                name: "metric::tests::compact_scope_index_matches_exhaustive_incident_selection",
                source: "crates/runmat-meshing/size/src/metric/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Compact scope lookup matches exhaustive incident selection across source groups and sparse incident sets. This preserves canonical selection; whole-stage allocation, cancellation and native scaling require separate evidence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-DIRECTIONAL-INTERSECTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-size",
                target: CargoTarget::Library,
                name: "metric::tests::tensor_intersection_preserves_each_direction_and_equal_cross_terms",
                source: "crates/runmat-meshing/size/src/metric/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Diagonal maxima with equal cross terms and conservative sums for distinct noncommuting tensors preserve sampled directional constraints and exact duplicate identity. This is not a minimum-volume anisotropic intersection or an arbitrary floating-point proof.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-IDEMPOTENT-GRADING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-size",
                target: CargoTarget::Library,
                name: "grading::indexed::tests::anisotropic_grading_preserves_directional_ratios_and_is_idempotent",
                source: "crates/runmat-meshing/size/src/grading/indexed/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A positive tensor scaling preserves every directional ratio while reaching the original finest-length cap, and repeated grading is unchanged. Isotropic cap realization remains exact. This component case does not establish spatial adaptation, arbitrary floating-point dominance or whole-study throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CONSTRAINT-STAR-PHYSICAL-ADMISSION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_mesh::constraint_star::tests::visible_nondelaunay_stars_decline_without_weakening_physical_admission",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/constraint_star/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "governed_path_equivalence",
                envelopes: KERNEL,
                boundary: "A visible conforming star is accepted only when independent physical Delaunay admission succeeds; visible but illegal stars decline the optimization. This kernel family does not independently prove every fallback recovery or governed native study.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-SIZED-CURVE-OWNERSHIP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "refinement::tests::sized_curve_boundary::sized_curve_allows_off_edge_size_refinement_but_retains_required_shared_splits",
                source: "crates/runmat-meshing/surface/src/refinement/tests/sized_curve_boundary.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Off-edge size-only refinement preserves a sizing-valid exact curve, including reversed uses. Geometric-error candidates, on-edge points and oversized protected edges still require shared splits. Final face quality and complete CAD conformity require independent checks.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BOUNDARY-REALIZATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "face_geometry::tests::realization::boundary_realization_preserves_original_surface_samples_and_geometric_acceptance",
                source: "crates/runmat-meshing/surface/src/face_geometry/tests/realization.rs",
            },
            claims: &[CoverageClaim {
                obligation: "governed_path_equivalence",
                envelopes: KERNEL,
                boundary: "Original shared curve coordinates define realized boundary nodes while exact surface samples remain unchanged. Independent admission measures their discrepancy against the original chordal envelope; fixed coordinates outside that envelope fail before futile interior insertion. This kernel witness does not establish native CAD or whole-study qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BOUNDARY-IDENTITY-ADMISSION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "face_geometry::tests::realization::boundary_coordinate_index_rejects_conflicting_shared_identities_and_denied_preparation",
                source: "crates/runmat-meshing/surface/src/face_geometry/tests/realization.rs",
            },
            claims: &[CoverageClaim {
                obligation: "shared_immutable_inventories",
                envelopes: KERNEL,
                boundary: "Conflicting coordinates for one original shared identity, cancelled preparation and denied allocation reject. Borrowed coordinate authority does not clone curve inventories; this is component admission evidence, not measured whole-worker memory.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BOUNDARY-INDEX-SCALING",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "face_geometry::tests::realization::large_borrowed_boundary_index_preserves_every_point_and_bounds_sort_preparation",
                source: "crates/runmat-meshing/surface/src/face_geometry/tests/realization.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A multi-run borrowed boundary index preserves every original point, charges pointer inventory and bounded cooperative sorting, and rejects denied search work. This finite fixture does not prove arbitrary native model throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SEPARATED-SIZE-CANDIDATES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "refinement::tests::separated_size_candidate::size_refinement_separates_the_midpoint_from_a_nonincident_node",
                source: "crates/runmat-meshing/surface/src/refinement/tests/separated_size_candidate.rs",
            },
            claims: &[CoverageClaim {
                obligation: "governed_path_equivalence",
                envelopes: KERNEL,
                boundary: "A size-refinement midpoint close to a nonincident vertex is replaced by a bounded separated point in the same triangle before boundary classification. An infeasible spacing request remains rejected; original final quality checks still apply.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-INDEPENDENT-SHAPE-EXTREMA",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::volume_quality::validation::comparison::extrema_tests::rejected_shape_extrema_compare_primitives_without_reciprocal_amplification",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/validation/comparison.rs",
            },
            claims: &[CoverageClaim {
                obligation: "governed_path_equivalence",
                envelopes: KERNEL,
                boundary: "Rejected nearly collapsed cells compare primitive Jacobians under the existing per-cell arithmetic bound instead of unstable reciprocal ranks. Wrong well-conditioned extrema, material primitive substitutions, nonfinite values and missing extrema reject. This does not admit a previously rejected physical cell.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SEPARATED-CANDIDATE-CANCELLATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "refinement::tests::separated_size_candidate::separated_candidate_observes_cancellation_inside_the_node_inventory",
                source: "crates/runmat-meshing/surface/src/refinement/tests/separated_size_candidate.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Candidate separation observes cancellation inside the original node inventory, retains only one best candidate, and propagates a typed failure before mutation. This component check does not measure arbitrary CAD throughput.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PROJECTION-METRIC-AUTHORITY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::solver_topology::tests::metric_authority::projection_preserves_the_original_volume_metric_without_rewriting_the_authored_request",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/tests/metric_authority.rs",
            },
            claims: &[CoverageClaim {
                obligation: "governed_path_equivalence",
                envelopes: KERNEL,
                boundary: "Solver projection validates the original volume against its original derived metric while preserving the authored request. Substitution with the authored base metric rejects. This kernel check does not establish whole native execution or a physical accuracy envelope.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-TET10-AFFINE-GEOMETRY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::solver_topology::order_elevation::affine::tests::affine_detection_respects_scale_and_retains_material_curvature",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/order_elevation/affine.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Floating-roundoff affine detection is invariant across the tested translations/scales and rejects material curvature and nonfinite coordinates. The production fast path retains every coordinate and requires independent positive-Jacobian certification first; this primitive test alone is not whole-study evidence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-TET10-SOURCE-QUERY-COMPLETENESS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::solver_topology::order_elevation::geometry::sources::spatial::tests::local_candidates_preserve_exhaustive_boundary_and_near_boundary_locations",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/order_elevation/geometry/sources/spatial/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Face/chart-local broad-phase queries preserve exhaustive barycentric locations on boundaries and near them across scale and translation. Original narrow-phase and parameter-image admission remain authoritative; this is finite component equivalence evidence.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-TET10-LOCAL-SOURCE-QUERIES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::solver_topology::order_elevation::geometry::sources::spatial::tests::localized_queries_do_not_scan_the_complete_surface_inventory_and_cancel",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/order_elevation/geometry/sources/spatial/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "A separated spatial inventory checks bounded local query work instead of complete-surface scans and observes cancellation inside tree traversal. Synthetic spatial copies are not accepted CAD, native throughput, or whole-mesh qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-TET10-SOURCE-ADMISSION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::solver_topology::order_elevation::geometry::sources::tests::prepared_sources_reject_unadmitted_storage_and_search_before_construction",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/order_elevation/geometry/sources/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Elevation source preparation rejects denied storage and search work before constructing borrowed lookup tables. Its conservative component storage envelope is not measured RSS or a complete worker-memory proof.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-TET10-SOURCE-IMAGE-OWNERSHIP",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-tetrahedron",
                target: CargoTarget::Library,
                name: "cdt::solver_topology::order_elevation::geometry::sources::tests::source_chart_images_reject_coedges_from_another_face_or_edge",
                source: "crates/runmat-meshing/tetrahedron/src/cdt/solver_topology/order_elevation/geometry/sources/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Prepared source images reject a present coedge identity belonging to another face or edge. All admitted periodic and singular images remain retained; this component ownership check does not replace original exact-geometry validation.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PORTABLE-ANALYTIC-LINE-TRIMS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-geometry-core",
                target: CargoTarget::Library,
                name: "model::exact_evaluator::portable::trim_classifier::line::tests::analytic_lines_classify_concave_boundaries_across_scale_and_orientation_without_allocation",
                source: "crates/runmat-geometry/core/src/model/exact_evaluator/portable/trim_classifier/line/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Straight portable trims use analytic projection and winding, matching the explicit concave-polygon inside/outside/boundary references across the tested scales and orientations with zero allocation budget. Curved trims keep the general method; this is finite portable-kernel evidence, not native CAD or whole-study qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PORTABLE-LINE-TRIM-FAILURES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-geometry-core",
                target: CargoTarget::Library,
                name: "model::exact_evaluator::portable::trim_classifier::line::tests::analytic_line_decisions_fall_back_near_the_tolerance_and_preserve_resource_failures",
                source: "crates/runmat-geometry/core/src/model/exact_evaluator/portable/trim_classifier/line/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Near-tolerance straight-edge decisions fall back to the existing general method. Search denial and cancellation remain typed failures; the fast path does not enlarge the boundary tolerance or suppress resource checks.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COLD-RECORD-EXACT-CAPACITY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-core",
                target: CargoTarget::Library,
                name: "contracts::canonical::chunk::closure::reconstruction::tests::fragmented_reconstruction_reserves_exact_output_and_preserves_closure",
                source: "crates/runmat-meshing/core/src/contracts/canonical/chunk/closure/reconstruction/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: "Across the tested chunk sizes and fragmented multi-stream records, preflight equals the actual output buffer capacities, admission denial stops reconstruction, and altered content identity rejects. This finite record-reconstruction check excludes imported typed metadata, constructor output, publication metadata, allocator overhead and whole-task memory qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COLD-RECORD-STAGE-BUDGET",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution",
                target: CargoTarget::Library,
                name: "controlled_streams::admission_tests::reconstructed_records_share_the_stage_ceiling_and_release_with_consumer_scope",
                source: "crates/runmat-meshing/execution/src/controlled_streams.rs",
            },
            claims: &[CoverageClaim {
                obligation: "phase_lifetimes",
                envelopes: KERNEL,
                boundary: "The finite cold-record fixture rejects zero and one-byte-short stage budgets with MemoryBudgetExceeded, accepts its exact requested reconstruction budget, preserves original records and releases the consumer scope completely. This does not establish a complete projection, original CAD study, native-host ceiling or process RSS bound.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COLD-INPUT-CONTROL-CLASS",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-execution", target: CargoTarget::Library, name: "serial::admission::tests::controlled::all_cold_input_classes_admit_each_storage_boundary_and_preserve_original_failures", source: "crates/runmat-meshing/execution/src/serial/admission/tests/controlled.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "All exact/faceted/evidence import allocation boundaries reject injected denial before storage allocation and retain original cancellation/deadline classes. Successful reads preserve every original provider buffer; scope completion releases declared input storage. Synthetic metadata is not original CAD, physics, whole-process RSS or native throughput evidence." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-GEOMETRY-VALIDATION-SCRATCH",
            locator: WitnessLocator::CargoTest { package: "runmat-geometry-core", target: CargoTarget::Library, name: "model::validation_storage::tests::exact_validation_releases_all_scratch_between_queries_and_rejects_short_admission", source: "crates/runmat-geometry/core/src/model/validation_storage/tests.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Exact topology and evaluator validators admit typed borrowed-index and claim scratch, accept their exact observed requested peak, reject a one-byte-short ceiling and release all temporary storage across repeated queries and interruption. This measures declared component storage, not allocator overhead or whole-study residency." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-FACETED-VALIDATION-SCALE",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-execution", target: CargoTarget::Library, name: "serial::admission::tests::controlled::scale::flat_validator_scratch_scales_linearly_releases_and_preserves_unordered_inputs", source: "crates/runmat-meshing/execution/src/serial/admission/tests/controlled/scale.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Analytic disconnected tetrahedra at 1/16/256 components have linear requested validation scratch, exact/short admission and complete release. Vertex/triangle/shell permutations preserve valid topology. This excludes native CAD meshing, accuracy of a numerical study and physical memory measurements." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-HEALING-REVISION-SCRATCH",
            locator: WitnessLocator::CargoTest { package: "runmat-geometry-core", target: CargoTarget::Library, name: "model::healing::tests::admission::healing_prepares_one_borrowed_revision_index_and_releases_it", source: "crates/runmat-geometry/core/src/model/healing/tests/admission.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "The finite healing report prepares borrowed revision membership once per validation; repeated calls retain no scratch and exact/short budgets and deadline interruption preserve cleanup. This is not independent geometric healing certification or a whole-model performance measurement." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ACTUAL-READ-ADMISSION",
            locator: WitnessLocator::CargoTest { package: "runmat-execution-artifact", target: CargoTarget::Library, name: "cache::filesystem::bounded_tests::admitted_reads_charge_actual_buffer_before_payload_and_preserve_hostile_bounds", source: "crates/runmat-execution-artifact/src/cache/filesystem/bounded_tests.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Filesystem reads admit the actual object buffer before allocation, within the caller bound. Denial, missing objects and poisoned bytes preserve bounded reads and content checks. This is declared buffer storage, not allocator overhead, measured process RSS or numerical qualification." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COLD-IDENTITY-READ",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-execution", target: CargoTarget::Library, name: "serial::admission::tests::cold_stage_identity_read_admits_actual_storage_instead_of_artifact_quota", source: "crates/runmat-meshing/execution/src/serial/admission/tests.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Cold stage identity, manifest and chunks retain original values with actual provider-buffer admission rather than reserving the complete artifact quota. Exact requested peak accepts, one byte less rejects and scope exit releases all storage. The fixture does not qualify original CAD, PDE accuracy or whole-task residency." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-BATCH-LOCALITY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "refinement::tests::local_queries::indexed_batches_preserve_direct_anisotropic_decisions_and_local_work",
                source: "crates/runmat-meshing/surface/src/refinement/tests/local_queries.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Indexed batching matches direct canonical decisions under the tested anisotropic metrics and bounds checkpoint work on 1,024 independent regions. This finite spatial-search witness does not establish original CAD convergence or whole-study performance." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-DOMAIN-LOCALITY",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "refinement::select::domain::tests::domain_index_preserves_direct_membership_and_local_search_work",
                source: "crates/runmat-meshing/surface/src/refinement/select/domain.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Indexed membership matches the direct robust orientation walk on interior, boundary and exterior points over 1,024 disconnected regions with bounded local work. This is finite domain-search evidence, not independent CAD or PDE qualification." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-QUERY-LIFETIME",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "refinement::tests::local_queries::query_scratch_is_admitted_before_allocation_and_released_after_use",
                source: "crates/runmat-meshing/surface/src/refinement/tests/local_queries.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Query construction admits input, sorting scratch and retained index storage before use; denial preserves its typed cause and success releases temporary storage. Declared storage excludes allocator overhead and measured RSS." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-QUERY-INTERRUPTION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "refinement::tests::local_queries::interrupted_query_preparation_releases_completed_indexes",
                source: "crates/runmat-meshing/surface/src/refinement/tests/local_queries.rs",
            },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Interruptions throughout query construction and use release previously admitted indexes and preserve cancellation. This finite ownership witness does not certify whole-task residency." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-MEMBERSHIP-EQUIVALENCE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "refinement::insertion::candidate::tests::indexed_membership_matches_direct_admission_in_disconnected_trimmed_regions",
                source: "crates/runmat-meshing/surface/src/refinement/insertion/candidate/tests.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Batch insertion admission preserves direct node duplication, protected-segment and trimmed-domain decisions on the tested disconnected regions. Existing robust predicates remain authoritative." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-INFLUENCE-ENCLOSURE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-surface",
                target: CargoTarget::Library,
                name: "refinement::encroachment::influence::tests::influence_enclosure_never_discards_direct_encroachment",
                source: "crates/runmat-meshing/surface/src/refinement/encroachment/influence.rs",
            },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Conservative segment-influence boxes retain direct rounded encroachment decisions across the tested anisotropy and translations. This enclosure witness does not replace complete surface quality acceptance." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PLANARITY-EXACT-EQUIVALENCE",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-surface", target: CargoTarget::Library, name: "exact_cdt::recovery::planarity::tests::indexed_planarity_matches_exhaustive_crossings_contacts_and_collinear_overlap", source: "crates/runmat-meshing/surface/src/exact_cdt/recovery/planarity/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Indexed edge-planarity decisions and successful predicate counts match exhaustive closed-bounds pair checks across the tested scales, translations, crossings, endpoint contacts and collinear overlaps. This finite witness does not establish every CAD topology or PDE qualification." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PLANARITY-WORK-AND-STORAGE",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-surface", target: CargoTarget::Library, name: "exact_cdt::recovery::planarity::tests::long_parallel_edges_have_local_work_and_exact_scratch_admission", source: "crates/runmat-meshing/surface/src/exact_cdt/recovery/planarity/tests.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "For 8,192 disjoint long edges sharing the same x interval, spatial traversal has bounded local work and preserves zero intersection predicates. Exact requested packed-index construction storage accepts, one byte less rejects, and temporary storage releases completely. The requested-buffer inventory excludes allocator overhead and process RSS." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PLANARITY-INTERRUPTION",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-surface", target: CargoTarget::Library, name: "exact_cdt::recovery::planarity::tests::interrupted_planarity_index_releases_admitted_storage", source: "crates/runmat-meshing/surface/src/exact_cdt/recovery/planarity/tests.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "An interruption during planarity index construction preserves cancellation and releases previously admitted scratch. This finite component fixture is not whole-task memory qualification." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PACKED-QUERY-FIRST-ERROR",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-core", target: CargoTarget::Library, name: "quality::spatial_index::packed::tests::fallible_packed_visitor_preserves_first_predicate_failure_without_buffering", source: "crates/runmat-meshing/core/src/quality/spatial_index/packed/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Streaming packed queries stop at the first caller predicate failure without a second candidate buffer; original errors remain observable." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-PLANARITY-FAILURE-TRANSPORT",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-execution", target: CargoTarget::Library, name: "surface_kernel::error::planarity_control_tests::surface_planarity_failures_preserve_resource_category_and_detail", source: "crates/runmat-meshing/execution/src/surface_kernel/error.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Surface-stage failure construction preserves memory denial and cancellation categories, operation and original diagnostic detail from controlled planarity validation." }],
        },
        Witness {
            id:"FEA-WITNESS-SHARED-SURFACE-GEOMETRY-BUFFERS",
            locator:WitnessLocator::CargoTest{package:"runmat-meshing-surface",target:CargoTarget::Library,name:"face_geometry::tests::storage::complete_geometry_buffers_admit_exact_requested_storage_and_reject_one_byte_less",source:"crates/runmat-meshing/surface/src/face_geometry/tests/storage.rs"},
            claims:&[CoverageClaim{obligation:"phase_lifetimes",envelopes:KERNEL,boundary:"Complete vertex and triangle geometry buffers are admitted before allocation. Exact requested contiguous capacity accepts, one byte less rejects, and scope exit releases those buffers. Shared provenance is separately field-owned; allocator headers and process RSS are excluded."}],
        },
        Witness {
            id:"FEA-WITNESS-SHARED-SURFACE-SHARED-PROVENANCE",
            locator:WitnessLocator::CargoTest{package:"runmat-meshing-surface",target:CargoTarget::Library,name:"exact_metric::tests::ownership::face_provenance_is_shared_without_weakening_independent_tamper_rejection",source:"crates/runmat-meshing/surface/src/exact_metric/tests/ownership.rs"},
            claims:&[CoverageClaim{obligation:"phase_lifetimes",envelopes:KERNEL,boundary:"Thousands of point evaluations share one immutable face identity and source-provenance allocation while retaining the same numerical values. Copy-on-write tampering rejects independently and leaves other evaluations intact. Sharing is not serialized proof authority."}],
        },
        Witness {
            id:"FEA-WITNESS-SHARED-SURFACE-PREPARED-FIELD-REUSE",
            locator:WitnessLocator::CargoTest{package:"runmat-meshing-surface",target:CargoTarget::Library,name:"face_pipeline_tests::prepared_metric::one_partition_reuses_one_prepared_producer_field_across_geometry_and_acceptance",source:"crates/runmat-meshing/surface/src/face_pipeline_tests/prepared_metric.rs"},
            claims:&[CoverageClaim{obligation:"bounded_local_search",envelopes:KERNEL,boundary:"A complete cold exact-face partition prepares its producer metric field once across geometry and acceptance. Independent validation reconstructs its own input-derived reference; this fixture does not establish complete original CAD convergence or whole-study scalability."}],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-ADMITTED-SAMPLING",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-execution", target: CargoTarget::Library, name: "surface_kernel::tests::admitted_surface_work_preserves_sampling_depth_and_honors_small_and_large_budgets", source: "crates/runmat-meshing/execution/src/surface_kernel/tests.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Production acceptance sample capacity follows the explicit stage search-work budget for small and dense meshes, preserving adaptive depths and quality margins, and rejecting recursion grants below the required depth. Actual exact evaluations remain charged to shared stage work. This finite policy witness does not establish whole-task timing or RSS." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-SAMPLING-PREFLIGHT",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-surface", target: CargoTarget::Library, name: "face_geometry::tests::acceptance_work::insufficient_adaptive_sample_budget_rejects_before_geometry_work", source: "crates/runmat-meshing/surface/src/face_geometry/tests/acceptance_work.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "An adaptive sampling demand exceeding the explicit cap rejects before independent geometry evaluation or report allocation. Overflow rejects conservatively. The lower bound never authorizes skipping required samples or deeper quality-driven refinement." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-INDEXED-ROUND-TRIP",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-surface", target: CargoTarget::Library, name: "face_pipeline_tests::publication_scale::indexed_surface_preserves_original_values_and_admits_before_decode", source: "crates/runmat-meshing/surface/src/face_pipeline_tests/publication_scale.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Lossless indexed surface transport preserves original identities, quality and signed-zero values, inventories typed reconstruction before decoding, and rejects denied admission. This finite representation fixture does not establish CAD or PDE acceptance, whole-task RSS, or arbitrary model capacity." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-STREAMED-LIFETIME",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-execution", target: CargoTarget::Library, name: "subtask_cache::tests::original_faces_reuse_across_partition_layouts_with_complete_boundary_identity", source: "crates/runmat-meshing/execution/src/subtask_cache/tests.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Fresh and cold restored faces preserve the original accepted values while streamed rich-face storage releases after successful or rejected consumption. Producer dictionary and decoder workspace remain charged through their actual use. This component witness does not close native publication, whole-join memory, or complete-study scalability." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-MOVED-JOIN-LINEAGE",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-surface", target: CargoTarget::Library, name: "face_pipeline_tests::complete_face_partition_runs_the_exact_surface_pipeline", source: "crates/runmat-meshing/surface/src/face_pipeline_tests.rs" },
            claims: &[CoverageClaim { obligation: "shared_immutable_inventories", envelopes: KERNEL, boundary: "Move-based joining reproduces the original decision and canonical bytes. Independent borrowed prerequisite checks reject substituted coordinates, quality, evaluator parameters, face coverage and bounds. Construction-only accepted results cannot be restored from serialized flags. This is a finite lineage witness, not whole-operator storage or performance qualification." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-KERNEL-RECORD-OWNERSHIP",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-execution", target: CargoTarget::Library, name: "geometry_control::tests::phase::publication_releases_kernel_scratch_and_drops_record_charge_at_last_use", source: "crates/runmat-meshing/execution/src/geometry_control/tests/phase.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "All kernel views share the stage ceiling. Imported inputs remain charged while obsolete kernel storage releases and admitted returned records retain their original capacity charge through their last use. This finite ownership witness does not establish model capacity or measured RSS." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-KERNEL-RECORD-REJECTION",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-execution", target: CargoTarget::Library, name: "geometry_control::tests::phase::unadmitted_record_transfer_rejects_and_keeps_ownership_until_error_cleanup", source: "crates/runmat-meshing/execution/src/geometry_control/tests/phase.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "A returned record inventory exceeding retained kernel admission rejects without discarding owned charges before error cleanup. This checks the stage ownership boundary, not every concrete encoder." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CURVE-CODEC-OWNERSHIP",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-curve", target: CargoTarget::Library, name: "shared::tests::codec_admission::admitted_curve_and_partition_encoders_preserve_bytes_and_release_storage", source: "crates/runmat-meshing/curve/src/shared/tests/codec_admission.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Curve and partition encoders preserve original canonical bytes, reject denied workspace before byte production, retain actual output capacity, and release temporary storage. The fixture does not establish native execution or arbitrary CAD capacity." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PROJECTION-COLD-ADMISSION",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-core", target: CargoTarget::Library, name: "contracts::canonical::codec::solver_projection::tests::cold_projection_admission_precedes_materialization_and_independent_validation", source: "crates/runmat-meshing/core/src/contracts/canonical/codec/solver_projection/tests.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Both element orders retain original cold projection values, separate typed materialization and topology scratch admission, preserve the exact denied callback, and reject canonical but forged topology. This witness does not establish whole-pipeline storage or model performance." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-IMMUTABLE-IMPORT-OWNERSHIP",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-execution", target: CargoTarget::Library, name: "serial::admission::tests::stage::immutable_stage_import_transfers_only_original_retention_and_preserves_denials", source: "crates/runmat-meshing/execution/src/serial/admission/tests/stage.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Complete immutable stage closures preserve their original objects and reference authority while verification scratch releases before the next input. Every tested allocation denial and sampled time/cancellation boundaries preserve their original cause and release owned charges. This finite fixture does not establish arbitrary closure capacity or whole-study performance." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-IMMUTABLE-CAPACITY-INVENTORY",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-execution", target: CargoTarget::Library, name: "publication::storage::tests::retained_publication_counts_spare_capacities_without_changing_canonical_content", source: "crates/runmat-meshing/execution/src/publication/storage.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Retained publication storage includes spare reference, object-payload, manifest and descriptor capacities without changing canonical content. Buffer capacities, encoded lengths and measured RSS remain separate quantities." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-DIRECTIONAL-REFINEMENT",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-size",
                target: CargoTarget::Library,
                name: "metric::subspace::tests::refinement_preserves_unrelated_axes_and_existing_constraints",
                source: "crates/runmat-meshing/size/src/metric/subspace.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Sampled directional quadratic forms preserve the input SPD constraint and unrelated orthogonal directions while satisfying the measured directional length. Invalid direction and length rejection has separate tests. This is a finite numerical kernel family, not an interval proof or a complete geometric sizing qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-CYLINDRICAL-DIRECTIONS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-curve",
                target: CargoTarget::Library,
                name: "shared::geometry_metric::directional_tests::cylindrical_curvature_refines_the_cross_section_without_refining_the_extrusion_axis",
                source: "crates/runmat-meshing/curve/src/shared/geometry_metric/directional_tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Portable exact cylindrical derivatives pass through occurrence transforms, principal physical directions and the production source derivation. Three translated/rotated placements preserve the flat extrusion direction, refine curved directions, and retain deterministic cold request identity. No complete CAD surface, volume, PDE or native study acceptance is established.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-THIN-GAP-DIRECTIONS",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-curve",
                target: CargoTarget::Library,
                name: "shared::geometry_metric::proximity::tests::thin_extrusion_gap_resolution_preserves_long_direction_under_rotation_and_translation",
                source: "crates/runmat-meshing/curve/src/shared/geometry_metric/proximity.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Three translated/rotated long-edge pairs retain half-separation resolution across a thin gap while preserving the long direction at the global baseline. Original contact and search-control cases remain separate. Sampled witnesses are not a closest-distance certificate or general local spatial-support qualification.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-BOUNDED-TANGENT-ENVELOPE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-size",
                target: CargoTarget::Library,
                name: "metric::subspace::tests::repeated_rotating_tangents_have_one_bounded_planar_envelope",
                source: "crates/runmat-meshing/size/src/metric/subspace.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Nine,33 and4097 rotating tangent constraints retain one planar envelope with at most three basis directions, without repeated density growth or refinement of the orthogonal axis. This finite numerical component test is not an interval proof, a spatially local sizing policy or original CAD/PDE acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-SOURCE-SET-BOUND",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-size", target: CargoTarget::Library, name: "metric::intersection::tests::thousands_of_noncommuting_sources_do_not_multiply_density_or_refine_the_common_normal", source: "crates/runmat-meshing/size/src/metric/intersection/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Nine,33 and4097 noncommuting constraints share a source-count-independent positive-subspace envelope; independent directional inequalities check each input and the common normal. Dominating original tensors remain exact. This is numerical sizing construction evidence, not an interval certificate, spatial localization or native CAD acceptance." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SERIALIZED-SHAPE-INTEGER-BOUND",
            locator: WitnessLocator::CargoTest { package: "runmat-canonical-codec", target: CargoTarget::Library, name: "serialized_profile::tests::shape::shape_bounds_integer_transitions_without_changing_exact_encoding_profiles", source: "crates/runmat-canonical-codec/src/serialized_profile/tests/shape.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Shape profiles reserve complete Serde integer value ranges across CBOR-width transitions while exact-value encoding profiles remain compact. Strings, collection counts, variants and optional branches remain the domain caller's envelope responsibility. This codec policy witness does not establish whole-task memory or numerical qualification." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COUPLED-QUALITY-ADMISSION",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_sliver::admission::tests::coupled_canonical_admission_uses_joint_order_while_standalone_preserves_refinement", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/admission/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Canonical coupled admission accepts a strictly lower combined size/shape defect and rejects a reverse trade under either primary source objective; the standalone shape treatment still preserves refinement. The reference comparator independently merges cell records. This is an algebraic policy witness, not a terminal physical-mesh or original-study qualification." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-COUPLED-QUALITY-PROGRESS",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_quality::working::ranking::tests::joint_delta_order_matches_complete_spectra_and_rejects_cross_phase_undo", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/working/ranking/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "A four-stream sparse-delta merge matches independently sorted complete size/shape spectra and rejects cross-phase undo. The coupled scheduler requires shared strict progress; standalone subdivision retains its separate shape-preservation policy. This ordering proof is not a guarantee that an arbitrary protected boundary admits the requested mesh quality." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-NUMERIC-CODEC-INVENTORY",
            locator: WitnessLocator::CargoTest { package: "runmat-analysis-fea", target: CargoTarget::Library, name: "planning::resource::numeric_profile::tests::count_only_inventory_covers_real_codec_headers_and_large_arrays", source: "crates/runmat-analysis/fea/src/planning/resource/numeric_profile.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "Count-only numerical profiling covers real codec arrays across header-width transitions and plans100million values without constructing or visiting that array. Domain metadata and transport remain separately reserved. This is structural codec accounting, not measured RSS or whole study admission." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MODAL-OUTPUT-PROFILE",
            locator: WitnessLocator::CargoTest { package: "runmat-analysis-fea", target: CargoTarget::Library, name: "planning::resource::modal::output::tests::full_search_capacity_does_not_treat_every_numeric_byte_as_a_map_node", source: "crates/runmat-analysis/fea/src/planning/resource/modal/output.rs" },
            claims: &[CoverageClaim { obligation: "phase_lifetimes", envelopes: KERNEL, boundary: "A60000DOF/256vector/3residual numeric output shape retains the complete search-capacity bound and conservative metadata while avoiding the unknown-shape byte multiplier. This count-only planning witness does not establish an eigensolve, physical convergence or whole-task capacity." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PROSPECTIVE-CHILD-METRIC",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_quality::working::context::tests::steiner_children_do_not_inherit_a_disappearing_boundary_scope", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/working/context/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Prospective Steiner children resolve their actual surviving boundary entities through the same metric authority as final evaluation. Losing a constrained face restores the declared global metric rather than retaining a parent constraint. This scope check does not establish mesh convergence." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PREPARED-STEINER-GEOMETRY",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_sliver::insertion::geometry::prepared::tests::split_probes_resolve_each_child_scope_once", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/insertion/geometry/prepared/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Prepared edge-star children match separately evaluated child geometry and resolve each child scope once across 128 repeated probes. The complete trial still rederives graded size and quality; this local-work check is not a whole-task memory or speedup claim." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PROTECTED-CAVITY-GROWTH",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_sliver::insertion::visibility::tests::protected_worst_face_does_not_hide_other_quality_expansion_faces", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/insertion/visibility/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "A four-cell cavity considers other poor interior faces even when its worst face is protected. The resulting cavity retains exactly the original outer facets. This expansion check does not guarantee arbitrary protected geometries admit a quality mesh." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CLOSED-CAVITY-VERTICES",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::mutation::closed_cavity::tests::closed_replacement_preserves_enclosed_vertices_before_commit", source: "crates/runmat-meshing/tetrahedron/src/cdt/mutation/closed_cavity/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "A closed replacement that preserves exterior facets but removes an enclosed vertex is rejected before commit. The original point star remains unchanged. Vertex deletion requires separate explicit authority; exterior-boundary equality alone is insufficient." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-JOINT-CAVITY-RELOCATION",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_sliver::working::tests::cavity_relocation::joint_cavity_relocation_crosses_the_old_star_and_replays_atomically", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/working/tests/cavity_relocation.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "An interior point crosses its old star in a joint point/cavity transaction. Independent canonical construction checks the complete result; failed replacement and dropped trials restore point/quality state, and a single grouped seed replays through current admission. All retained points participate. This local transaction witness is not whole-model convergence or measured memory evidence." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SELECTED-DEFECT-PROGRESS",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_quality::working::ranking::tests::selected_class_progress_rejects_tail_only_stagnation_and_accepts_class_reduction", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/working/ranking/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "A complete-spectrum improvement confined below the selected defect class cannot discharge a repair pass. Reducing that class's multiplicity or priority is admitted; creating a higher defect is rejected. Shared sparse delta comparison needs no copied full spectrum. Physical acceptance still checks every unquantized bound independently." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-NEIGHBORING-POINT-TRANSACTION",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_sliver::working::tests::cavity_relocation::neighboring_points_move_and_replay_in_one_atomic_cavity_transaction", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/working/tests/cavity_relocation.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Neighboring interior points share one point/cavity transaction and one replay record. A canonical construction independently checks resulting connectivity and complete quality; failed and dropped candidates restore the original state. This checks transaction authority, not whole-model convergence or throughput." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-METRIC-CORNER-DERIVATIVES",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_quality::jacobian::gradient::tests::corner_derivatives_match_independent_metric_differences_across_orders_and_scales", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_quality/jacobian/gradient/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Analytic per-corner derivatives agree with independent physical-coordinate metric finite differences across point ordering, translation, scale and off-diagonal anisotropy. Degenerate and nonfinite inputs reject derivative proposals. Exact orientation and full quality remain separate acceptance authorities; these arithmetic checks establish no whole-model convergence or throughput." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-LOCAL-MAXIMIN-MODEL",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_sliver::relocation::geometry::block::ascent::tests::linear_maximin_bundle_accounts_for_nearby_constraints_and_stationarity", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/relocation/geometry/block/ascent.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "The local linear maximin proposal includes constraint slack, improves competing affine constraints within its radius, rejects stationary or fixed bottlenecks, and preserves common scaling. The model supplies proposals only; independent complete geometry and current scoped quality decide every accepted edit. This witness does not assert original-study acceptance or measured speedup." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-PREPARED-NEIGHBORING-BLOCK",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_sliver::relocation::geometry::block::tests::prepared_neighboring_blocks_escape_the_symmetric_minimax_trap_and_preserve_cpu_order", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/relocation/geometry/block/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "A symmetric two-point block improves the minimum incident shape quality while preserving exact positive orientation. Metric resolution occurs once per local cell, and one/four reusable CPU grants produce identical proposals. Numerical probes use prepared local geometry; this does not establish original-study acceptance or whole-task speedup." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-INCREMENTAL-CAVITY-FRONTIER",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_sliver::insertion::visibility::growth::tests::incremental_visibility_matches_scan_reference_and_resolves_each_face_metric_once", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/insertion/visibility/growth/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Incremental visibility and quality growth retain the independent scan reference's exact cavity under isotropic and off-diagonal metrics. Each fixed-point face metric is resolved once; the scan reference performs repeated queries. Canonical decision order, protected boundary and shared smoothing checks remain separate. This is an algorithmic work witness, not whole-model throughput or study acceptance." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CONFORMING-POINT-SUPPORT",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::insertion::conforming::tests::conforming_cell_face_and_edge_insertions_preserve_the_exact_outer_boundary", source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/conforming/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Explicit conforming insertion splits the complete containing cell, internal face or edge support, preserves the exact outer boundary, retains every point and passes independent canonical admission. A protected boundary split rejects without changing the original state. This contract does not assert Delaunay empty-sphere quality or original CAD convergence." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-CONFORMING-REFILL-INTENT",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::insertion::conforming::tests::conforming_refill_keeps_an_unaffected_tetrahedron_inside_the_euclidean_conflict_sphere", source: "crates/runmat-meshing/tetrahedron/src/cdt/insertion/conforming/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Exact predicates establish a point outside a chosen tetrahedron but inside its Euclidean circumsphere. Explicit conforming insertion retains that unaffected tetrahedron and passes independent physical topology admission. Quality refill therefore does not silently impose Euclidean conflict growth; callers requesting Delaunay insertion retain their separate operator. This is not a whole-study accuracy or throughput claim." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-EXPLICIT-CAVITY-POINT-MOTION",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_sliver::insertion::refill::tests::local_refill_moves_only_explicitly_named_enclosed_points_and_preserves_the_boundary", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/insertion/refill/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Local refill moves only explicitly named original enclosed points, preserves the other points and exact cavity boundary, and passes independent canonical construction with every point incident. Duplicate names and protected boundary point names reject. This checks refill authority, not convergence of an original CAD model." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-LOCAL-CAVITY-REFILL",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_sliver::insertion::refill::tests::local_refill_preserves_enclosed_points_and_the_original_cavity_boundary", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/insertion/refill/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "A local cone refill reinserts enclosed original points through the constrained insertion kernel, retains their coordinates and identities, leaves no orphaned vertex, and preserves the exact cavity boundary. Only cavity data is materialized. This fixture is not a whole-task memory measurement or convergence guarantee." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-ACTUAL-RECONNECTION-RANKING",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_sliver::reconnection::working::tests::coupled_direct_reconnection_selects_the_best_actual_cell_spectrum", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_sliver/reconnection/working/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Coupled direct reconnection is compared with complete-topology quality arithmetic across scoped-metric candidate cells and identity permutations. Actual changed-cell spectra decide selection; source-tensor proposal order supplies no final ranking authority. Standalone first-admissible behavior remains a separate contract." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-STALLED-STATE-OBSERVATION",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_mesh::quality_replay::tests::stalled::stalled_state_reaches_the_host_observer_without_replacing_the_verdict", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_replay/tests/stalled.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "An immutable, rejected tetrahedron reaches a host-owned snapshot observer. An injected optional observer failure preserves the original quality verdict; cold snapshot decoding preserves bound controls and remains rejected. No accepted mesh or continuation authority is created, and native cache retention under every budget is not asserted." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BOUNDARY-SUPPORT-CAPACITY",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-surface", target: CargoTarget::Library, name: "refinement::boundary_support::storage::tests::boundary_capacity_is_reserved_before_allocation_and_cannot_grow_past_the_boundary", source: "crates/runmat-meshing/surface/src/refinement/boundary_support/storage.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Actual vector capacity fits the admitted boundary-segment count for first and repeated insertions. A one-byte shortage rejects before allocation; exceeding the count cannot grow storage. This establishes typed scratch capacity admission, excluding allocator headers and whole-process RSS, not whole-model throughput or scientific acceptance." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-EDGE-LOCAL-SURFACE-SUPPORT",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-surface", target: CargoTarget::Library, name: "exact_metric::resolved::boundary::tests::boundary_support_preserves_edge_local_tensors_instead_of_using_the_face_envelope", source: "crates/runmat-meshing/surface/src/exact_metric/resolved/boundary/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "Prepared surface support retains each edge/region tensor separately from the combined face envelope, without modifying the original request. Interior support proposals pass shared exact insertion and spacing; whole workflow and original CAD convergence require separate evidence. This witness does not prove universal boundary feasibility or whole-task memory demand." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-RETAINED-TERMINAL-COLLAR",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_mesh::quality_replay::tests::retained_terminal_collar_requires_surface_remeshing", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_replay/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "An outward common-metric edge-star bound proves that the retained108-node/292-cell fixed boundary cannot satisfy its unchanged0.01 metric scaled-Jacobian target. Original and explicitly rebound million-operation controls reject with a surface-remeshing diagnosis before volume search. This is incompatibility classification, not positive convergence, surface repair, native CAD acceptance or study qualification." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SCOPED-COLLAR-QUALITY",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_mesh::quality_replay::tests::thin_boundary_collar_requires_surface_remeshing", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_replay/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "The retained96-node/236-cell scoped collar has a certified fixed-boundary quality conflict at the unchanged0.01 target. Increasing the explicitly bound search cap cannot make it feasible. Exact protected-facet preservation remains mandatory; a generated surface must instead be repaired by its surface owner. This witness does not establish positive convergence, upstream surface repair or completed studies." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-QUALITY-EXPLICIT-SEARCH-CAP",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-tetrahedron", target: CargoTarget::Library, name: "cdt::volume_mesh::quality_replay::tests::noncommuting_source_fixture_preserves_its_explicit_small_search_cap", source: "crates/runmat-meshing/tetrahedron/src/cdt/volume_mesh/quality_replay/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "A retained96node/236cell diagnostic binds an explicit2048candidate-per-pass cap and rejects substitution of a million-candidate policy before execution. Native governing requests are independently projected and charged by the shared stage authority; this authority check does not predict budget exhaustion or establish quality convergence, CAD acceptance or a completed study." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-GOVERNED-QUALITY-SEARCH",
            locator: WitnessLocator::CargoTest { package: "runmat-meshing-execution", target: CargoTarget::Library, name: "volume_kernel::tests::governed_optimization_uses_the_declared_search_envelope_without_a_standalone_cap", source: "crates/runmat-meshing/execution/src/volume_kernel/tests.rs" },
            claims: &[CoverageClaim { obligation: "bounded_local_search", envelopes: KERNEL, boundary: "The governed request projects its exact declared search envelope at small, standalone-default and million-operation sizes. Shared cumulative stage accounting still bounds every strategy. This contract check establishes neither quality convergence nor original mesh acceptance." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MODAL-CPU-GRANT",
            locator: WitnessLocator::CargoTest { package: "runmat-analysis-fea", target: CargoTarget::Library, name: "planning::resource::modal_tests::allocation::modal_cpu_grant_binds_identity_pool_storage_and_unchanged_accuracy_ceiling", source: "crates/runmat-analysis/fea/src/planning/resource/modal_tests/allocation.rs" },
            claims: &[CoverageClaim { obligation: "governed_path_equivalence", envelopes: KERNEL, boundary: "Serial and four-thread modal grants bind distinct resource identities and pool-stack storage while preserving the original accuracy ceiling. Native extraction uses the admitted reusable CPU pool; this witness does not assert parallel assembly or an arbitrary speedup." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-MODAL-NATIVE-PROFILE-REVALIDATION",
            locator: WitnessLocator::CargoTest { package: "runmat-execution-runner-native", target: CargoTarget::Library, name: "analysis_kernel::modal_tests::native_scalar_modal_preserves_cold_basis_receipts_fields_and_rejects_missing_source_budget", source: "crates/runmat-execution-runner-native/src/analysis_kernel/modal_tests.rs" },
            claims: &[CoverageClaim { obligation: "governed_path_equivalence", envelopes: KERNEL, boundary: "Tet4/Tet10 native scalar extraction, cold basis/receipt recovery and actual result codec profiles check the shared output reservation. One-byte construction understatement rejects. This finite workflow does not qualify arbitrary modal counts, realistic original studies or measured peak memory." }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BOUNDARY-FACE-TRACE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-curve",
                target: CargoTarget::Library,
                name: "shared::resolved_metric::tests::curve_boundary_trace_satisfies_every_adjacent_fine_face_before_surface_execution",
                source: "crates/runmat-meshing/curve/src/shared/resolved_metric/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Every boundary edge satisfies the shared incident face metric trace under three occurrence transforms, with stable repeated queries and source provenance. This component fixture does not establish complete native surface recovery or original study acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-BOUNDARY-NORMALIZATION",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution",
                target: CargoTarget::Library,
                name: "surface_kernel::tests::boundary_curve_target_honors_the_shared_surface_normalization_and_stricter_user_bounds",
                source: "crates/runmat-meshing/execution/src/surface_kernel/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "bounded_local_search",
                envelopes: KERNEL,
                boundary: "Curve preparation honors the shared unit metric target and stricter authored upper bounds without mutating the request. Actual surface quality and recovery remain independently checked.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-TERMINAL-SHARED-USAGE",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution",
                target: CargoTarget::Library,
                name: "budget::tests::terminal_evidence_retains_shared_usage_after_the_last_algorithm_checkpoint",
                source: "crates/runmat-meshing/execution/src/budget/tests.rs",
            },
            claims: &[CoverageClaim {
                obligation: "governed_path_equivalence",
                envelopes: KERNEL,
                boundary: "Terminal stage evidence retains cumulative shared search and iteration work and peak allocation after the last algorithm checkpoint, without adding duplicate absolute counters or resetting usage. This is accounting evidence, not measured RSS or whole study acceptance.",
            }],
        },
        Witness {
            id: "FEA-WITNESS-SHARED-SURFACE-JOIN-CONTROL-CAUSES",
            locator: WitnessLocator::CargoTest {
                package: "runmat-meshing-execution",
                target: CargoTarget::Library,
                name: "surface_kernel::error::planarity_control_tests::joined_acceptance_preserves_nested_geometry_control_causes",
                source: "crates/runmat-meshing/execution/src/surface_kernel/error.rs",
            },
            claims: &[CoverageClaim {
                obligation: "governed_path_equivalence",
                envelopes: KERNEL,
                boundary: "Joined acceptance preserves cancellation, time, memory, search, iteration and numerical-result categories through both nested metric paths, retaining the original diagnostic detail. This category transport fixture does not establish successful convergence, budget sufficiency or whole-study execution.",
            }],
        },
    ],
};
