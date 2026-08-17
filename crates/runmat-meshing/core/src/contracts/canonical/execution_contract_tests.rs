use std::collections::BTreeMap;

use super::*;

fn digest(byte: u8) -> StableDigest {
    StableDigest::from_bytes([byte; 32])
}

fn entity(kind: PersistentEntityKind, id: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: id.into(),
        assembly_path: vec!["root".into()],
    }
}

pub(super) fn batch_partition() -> MeshingPartitionDescriptor {
    MeshingPartitionDescriptor {
        kind: MeshingPartitionKind::CanonicalEntityBatch,
        partition_index: 0,
        partition_count: 2,
        entity_range: Some(CanonicalEntityRange {
            first: entity(PersistentEntityKind::Face, "face:001"),
            last: entity(PersistentEntityKind::Face, "face:016"),
            entity_count: 16,
        }),
    }
}

pub(super) fn stage_identity() -> MeshingStageIdentity {
    MeshingStageIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage: MeshingStageKind::SurfaceMesh,
        geometry: GeometryRevisionRef {
            source_digest: digest(1),
            geometry_revision: 3,
            persistent_mapping_version: 2,
        },
        resolved_request_digest: digest(2),
        tolerance_policy_digest: digest(3),
        metric_policy_digest: digest(4),
        algorithm_set_digest: digest(5),
        deterministic_seed: 17,
        prerequisite_artifact_digests: vec![digest(6), digest(7)],
        capability_cohort: Some("native-exact-cad-v1".into()),
    }
}

pub(super) fn workload() -> MeshingWorkloadRequest {
    MeshingWorkloadRequest {
        schema_version: MESHING_WORKLOAD_SCHEMA_VERSION,
        stage: MeshingStageKind::SurfaceMesh,
        stage_identity_digest: digest(8),
        partition: batch_partition(),
        input_manifest_digests: vec![digest(9)],
        required_capabilities: vec![
            MeshingCapabilityRequirement::HostWorkload {
                abi: "meshing-host-v2".into(),
            },
            MeshingCapabilityRequirement::ExactCadKernel {
                abi: "opencascade-7.9".into(),
            },
            MeshingCapabilityRequirement::MeshingAlgorithm {
                version: "surface-cdt-v2".into(),
            },
            MeshingCapabilityRequirement::ElementOrder {
                order: ElementOrder::Tet10,
            },
            MeshingCapabilityRequirement::DeterministicPlatformCohort {
                cohort: "native-exact-cad-v1".into(),
            },
        ],
    }
}

pub(super) fn progress(sequence: u64, completed_work: u64) -> MeshingProgress {
    MeshingProgress {
        schema_version: MESHING_PROGRESS_SCHEMA_VERSION,
        stage: MeshingStageKind::SurfaceMesh,
        partition_index: 0,
        sequence,
        completed_work,
        estimated_work: 16,
        entity_counts: BTreeMap::from([("faces_completed".into(), completed_work)]),
        peak_memory_bytes: 1_000 + completed_work,
        elapsed_time_ms: 10 + completed_work,
        consumed_search_work: completed_work * 2,
        cancellation_checkpoint: completed_work,
    }
}

#[test]
fn canonical_stage_partition_and_join_identities_round_trip() {
    let stage = stage_identity();
    stage.validate().unwrap();
    let encoded = serde_json::to_vec(&stage).unwrap();
    assert_eq!(
        serde_json::from_slice::<MeshingStageIdentity>(&encoded).unwrap(),
        stage
    );

    let partition = MeshingPartitionIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage_identity_digest: digest(10),
        partition: batch_partition(),
    };
    partition.validate().unwrap();

    let join = MeshingJoinIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage_identity_digest: digest(10),
        join_algorithm_version: "surface-stitch/v2".into(),
        ordered_partition_results: vec![
            MeshingPartitionResultRef {
                partition_index: 0,
                result_digest: digest(11),
            },
            MeshingPartitionResultRef {
                partition_index: 1,
                result_digest: digest(12),
            },
        ],
    };
    join.validate().unwrap();

    MeshingStageResultIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage: MeshingStageKind::SurfaceMesh,
        result_kind: MeshingStageResultKind::DeterministicJoin,
        producer_identity_digest: digest(13),
        logical_content_digest: digest(14),
        logical_entity_count: 16,
        invariant_summary_digest: digest(16),
    }
    .validate()
    .unwrap();

    MeshingValidationIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        subject_stage_result_digest: digest(17),
        geometry: stage.geometry.clone(),
        resolved_request_digest: stage.resolved_request_digest,
        validation_algorithm_version: "independent-validation/v2".into(),
        capability_cohort: stage.capability_cohort.clone(),
    }
    .validate()
    .unwrap();

    let mut reordered = join;
    reordered.ordered_partition_results.swap(0, 1);
    assert_eq!(
        reordered.validate().unwrap_err().field,
        "join partition results"
    );
}

#[test]
fn identities_reject_completion_order_and_physical_host_fields() {
    let mut value = serde_json::to_value(stage_identity()).unwrap();
    value["worker_id"] = serde_json::json!("worker-7");
    value["physical_path"] = serde_json::json!("/tmp/mesh");
    value["completed_at"] = serde_json::json!(1234);
    assert!(serde_json::from_value::<MeshingStageIdentity>(value).is_err());

    let mut invalid = stage_identity();
    invalid.prerequisite_artifact_digests.swap(0, 1);
    assert_eq!(
        invalid.validate().unwrap_err().field,
        "prerequisite artifact digests"
    );
}

#[test]
fn workload_and_manifest_reject_unknown_control_plane_fields() {
    let mut workload = serde_json::to_value(workload()).unwrap();
    workload["retry_policy"] = serde_json::json!("infrastructure");
    workload["worker_id"] = serde_json::json!("worker-9");
    assert!(serde_json::from_value::<MeshingWorkloadRequest>(workload).is_err());

    let digest_one = vec![1_u8; 32];
    let digest_two = vec![2_u8; 32];
    let mut manifest = serde_json::json!({
        "schema_version": MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
        "stage": "surface_mesh",
        "result_kind": "whole_stage",
        "logical_result_identity": digest_one,
        "disposition": "diagnostic_only",
        "prerequisite_manifest_digests": [],
        "invariant_summary_digest": digest_two,
        "chunks": [],
        "total_encoded_length": 0,
    });
    manifest["physical_path"] = serde_json::json!("/tmp/result");
    assert!(serde_json::from_value::<MeshingStageManifest>(manifest).is_err());
}

#[test]
fn stage_manifest_closes_over_ordered_typed_chunks() {
    let manifest = MeshingStageManifest {
        schema_version: MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
        stage: MeshingStageKind::SurfaceMesh,
        result_kind: MeshingStageResultKind::DeterministicJoin,
        logical_result_identity: digest(20),
        disposition: MeshingManifestDisposition::ValidatedDependency,
        prerequisite_manifest_digests: vec![digest(21)],
        invariant_summary_digest: digest(22),
        chunks: vec![
            MeshingChunkDescriptor {
                ordinal: 0,
                first_logical_entity_ordinal: 0,
                digest: digest(23),
                media_type: MeshingChunkMediaType::SurfacePartitions,
                schema_version: 2,
                encoded_length: 400,
                decoded_length: 800,
                logical_entity_count: 16,
            },
            MeshingChunkDescriptor {
                ordinal: 1,
                first_logical_entity_ordinal: 16,
                digest: digest(24),
                media_type: MeshingChunkMediaType::ValidationEvidence,
                schema_version: 2,
                encoded_length: 100,
                decoded_length: 120,
                logical_entity_count: 16,
            },
        ],
        total_encoded_length: 500,
    };
    manifest.validate().unwrap();
    assert!(manifest.is_dependency_eligible());
    assert_eq!(
        manifest.chunks[0].media_type.media_type(),
        "application/vnd.runmat.mesh-surfaces.v2"
    );

    let mut corrupt = manifest.clone();
    corrupt.chunks[1].ordinal = 0;
    assert_eq!(
        corrupt.validate().unwrap_err().field,
        "meshing chunk descriptor"
    );

    let mut diagnostic = manifest;
    diagnostic.disposition = MeshingManifestDisposition::DiagnosticOnly;
    diagnostic.validate().unwrap();
    assert!(!diagnostic.is_dependency_eligible());
}

#[test]
fn workload_requires_canonical_domain_capabilities_and_limits_entity_batching() {
    let request = workload();
    request.validate().unwrap();
    MeshingWorkloadResult::Validated {
        stage_manifest_digest: digest(30),
    }
    .validate_against(&request)
    .unwrap();

    let mut incompatible = request.clone();
    incompatible.required_capabilities.swap(0, 1);
    assert_eq!(
        incompatible.validate().unwrap_err().field,
        "meshing workload capabilities"
    );

    let mut invalid_partition = request;
    invalid_partition.stage = MeshingStageKind::Tetrahedralization;
    assert_eq!(
        invalid_partition.validate().unwrap_err().field,
        "meshing workload partition"
    );
}

#[test]
fn detailed_progress_is_bounded_and_monotone_per_partition() {
    let first = progress(1, 4);
    let next = progress(2, 9);
    next.validate_after(&first).unwrap();

    let mut regression = next;
    regression.completed_work = 3;
    assert_eq!(
        regression.validate_after(&first).unwrap_err().field,
        "meshing progress transition"
    );
}

#[test]
fn workload_failure_must_match_the_requested_stage() {
    let request = workload();
    let result = MeshingWorkloadResult::Failed {
        failure: MeshingFailure {
            schema_version: MESHING_FAILURE_SCHEMA_VERSION,
            category: MeshingFailureCategory::NumericalFailure,
            stage: MeshingStageKind::Optimization,
            operation: MeshingOperation::Optimize,
            entity_ids: Vec::new(),
            witnesses: Vec::new(),
            request_values: Vec::new(),
            achieved_values: Vec::new(),
            remediation: "inspect the failed optimization cavity".into(),
        },
    };
    assert_eq!(
        result.validate_against(&request).unwrap_err().field,
        "workload failure stage"
    );
}
