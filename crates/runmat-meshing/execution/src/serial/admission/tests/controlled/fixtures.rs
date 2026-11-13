use super::*;
use runmat_execution::value::ValueRef;
use runmat_execution_artifact::{object::ObjectInventoryLimits, LogicalObject};
use runmat_geometry_core::GeometryDocument;
use runmat_meshing_core::*;
use std::collections::BTreeMap;
pub(super) struct Fixture {
    pub document: Option<GeometryDocument>,
    pub root: ValueRef,
    pub objects: Vec<LogicalObject>,
    pub access: crate::ArtifactAccess,
}
impl Fixture {
    pub fn new(kind: Kind) -> Self {
        let access = crate::ArtifactAccess {
            authorization_scope: "cold-input-owner".into(),
            encryption_context: Digest::sha256(b"cold-input-owner"),
        };
        let limits = ObjectInventoryLimits::default();
        match kind {
            Kind::Exact => {
                let (document, topology, evaluators) = runmat_geometry_fixtures::exact_circle();
                let objects = crate::prepare_exact_geometry_objects(
                    document, topology, evaluators, None, None, limits,
                )
                .unwrap();
                let input =
                    crate::prepare_exact_geometry_input(objects, access.clone(), limits).unwrap();
                Self {
                    document: Some(input.geometry_objects().document.clone()),
                    root: input.root_input().clone(),
                    objects: input.geometry_objects().objects.clone(),
                    access,
                }
            }
            Kind::Faceted => {
                let (document, solid) = runmat_geometry_fixtures::faceted_tetrahedron();
                let objects =
                    crate::prepare_faceted_geometry_objects(document, solid, limits).unwrap();
                let input =
                    crate::prepare_faceted_geometry_input(objects, access.clone(), limits).unwrap();
                Self {
                    document: Some(input.geometry_objects().document.clone()),
                    root: input.root_input().clone(),
                    objects: input.geometry_objects().objects.clone(),
                    access,
                }
            }
            Kind::Evidence => {
                let objects = crate::prepare_evidence_objects(evidence(), limits).unwrap();
                let input = crate::prepare_evidence_input(objects, access.clone(), limits).unwrap();
                Self {
                    document: None,
                    root: input.root_input().clone(),
                    objects: input.evidence_objects().objects.clone(),
                    access,
                }
            }
        }
    }
}
fn evidence() -> MeshingEvidence {
    // Metadata-only observations: no claim that this fixture ran a mesher.
    let mut stage = MeshingStageEvidence {
        reused_from: None,
        stage: MeshingStageKind::Serialization,
        partition: MeshingPartitionDescriptor {
            kind: MeshingPartitionKind::WholeStage,
            partition_index: 0,
            partition_count: 1,
            entity_range: None,
        },
        stage_result_digest: StableDigest::from_bytes([9; 32]),
        entity_counts: BTreeMap::from([("accepted".into(), 1)]),
        invariants: vec![InvariantEvidence {
            invariant_id: "fixture_complete".into(),
            passed: true,
            measured: Some(1.0),
            required: Some(1.0),
            unit: Some("count".into()),
        }],
        achieved_error_distributions: BTreeMap::new(),
        completed_work: 1,
        estimated_work: 1,
        peak_memory_bytes: 100,
        peak_scratch_bytes: 10,
        search_work: 1,
        maximum_recursion_depth: 1,
        iterations: 1,
        elapsed_time_ms: 1,
        cancellation_checkpoints: 1,
    };
    stage.reused_from = Some(ReusedMeshingStageObservation {
        original_observation_digest: stage.canonical_digest().unwrap(),
        origin: runmat_meshing_core::ReusedMeshingStageOrigin::ImplementationCache {
            implementation_digest: StableDigest::from_bytes([8; 32]),
        },
    });
    MeshingEvidence {
        schema_version: MESHING_EVIDENCE_SCHEMA_VERSION,
        ceilings: crate::serial_tests::ceilings(),
        geometry: GeometryRevisionRef {
            source_unit_evidence:
                runmat_geometry_core::physical::LengthUnitEvidence::explicit_fallback(
                    runmat_geometry_core::LengthUnit::Meter,
                    "canonical solver fixture",
                ),
            source_digest: StableDigest::from_bytes([1; 32]),
            geometry_revision: 1,
            persistent_mapping_version: 1,
        },
        resolved_request_digest: StableDigest::from_bytes([2; 32]),
        artifact_digest: StableDigest::from_bytes([3; 32]),
        algorithms: AlgorithmVersionSet {
            geometry: "fixture/1".into(),
            curve: "fixture/1".into(),
            surface: "fixture/1".into(),
            plc: "fixture/1".into(),
            tetrahedron: "fixture/1".into(),
            optimization: "fixture/1".into(),
            validation: "fixture/1".into(),
        },
        deterministic_seed: 1,
        platform: PlatformBuildIdentity {
            capability_cohort: "fixture".into(),
            target_triple: "fixture".into(),
            build_digest: StableDigest::from_bytes([4; 32]),
            exact_kernel_abi: None,
        },
        stages: vec![stage],
        sizing: vec![],
        resources: MeshingResourceUsage {
            generated_nodes: 4,
            generated_elements: 1,
            peak_memory_bytes: 100,
            peak_scratch_bytes: 10,
            wall_time_ms: 1,
            artifact_bytes: 1,
            search_work: 1,
            maximum_recursion_depth: 1,
            iterations: 1,
        },
        cache_admission: CacheAdmissionDecision::Admitted,
    }
}
