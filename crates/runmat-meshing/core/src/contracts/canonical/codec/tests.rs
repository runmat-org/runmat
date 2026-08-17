use std::collections::BTreeMap;

use minicbor::Encoder;

use super::{CanonicalMeshingContract, CODEC_PREFIX};
use crate::contracts::canonical::artifact_tests::{artifact, evidence, request};
use crate::contracts::canonical::execution_contract_tests::{
    batch_partition, progress, stage_identity, workload,
};
use crate::contracts::canonical::{
    GeometryTolerancePolicy, MeshingChunkDescriptor, MeshingChunkMediaType, MeshingDiagnosticValue,
    MeshingFailure, MeshingFailureCategory, MeshingJoinIdentity, MeshingManifestDisposition,
    MeshingOperation, MeshingPartitionIdentity, MeshingPartitionResultRef, MeshingProgress,
    MeshingRequest, MeshingStageIdentity, MeshingStageKind, MeshingStageManifest,
    MeshingStageResultIdentity, MeshingStageResultKind, MeshingValidationIdentity,
    MeshingWorkloadResult, SolverMeshArtifact, StableDigest, MESHING_FAILURE_SCHEMA_VERSION,
    MESHING_IDENTITY_SCHEMA_VERSION, MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
};

fn digest(byte: u8) -> StableDigest {
    StableDigest::from_bytes([byte; 32])
}

#[test]
fn canonical_request_has_a_stable_golden_identity() {
    let request = request();
    let encoded = request.canonical_encode().unwrap();
    let decoded = MeshingRequest::canonical_decode(&encoded).unwrap();
    assert_eq!(decoded, request);
    assert_eq!(decoded.canonical_encode().unwrap(), encoded);
    assert_eq!(
        request.canonical_digest().unwrap().bytes(),
        &[
            118, 31, 129, 193, 177, 198, 141, 21, 177, 158, 5, 145, 157, 201, 176, 175, 13, 187,
            129, 95, 197, 255, 44, 16, 208, 239, 239, 179, 94, 117, 149, 8,
        ]
    );
}

#[test]
fn canonical_map_identity_is_independent_of_insertion_order() {
    let mut left = progress(2, 9);
    left.entity_counts = BTreeMap::from([
        ("edges_completed".into(), 12),
        ("faces_completed".into(), 9),
    ]);
    let mut right = progress(2, 9);
    right.entity_counts.clear();
    right.entity_counts.insert("faces_completed".into(), 9);
    right.entity_counts.insert("edges_completed".into(), 12);

    assert_eq!(
        left.canonical_encode().unwrap(),
        right.canonical_encode().unwrap()
    );
    assert_eq!(
        left.canonical_digest().unwrap(),
        right.canonical_digest().unwrap()
    );
}

#[test]
fn domains_prevent_equal_values_from_cross_contract_decoding() {
    let encoded = request().canonical_encode().unwrap();
    let error = MeshingStageIdentity::canonical_decode(&encoded).unwrap_err();
    assert_eq!(error.field, "canonical decoding domain");

    let tolerance = GeometryTolerancePolicy {
        source_tolerance_m: 1.0e-8,
        absolute_floor_m: 1.0e-10,
        model_relative_term: 1.0e-9,
        requested_deviation_m: 1.0e-5,
        maximum_healing_displacement_m: 1.0e-6,
    };
    assert_ne!(
        tolerance.canonical_digest().unwrap(),
        request().canonical_digest().unwrap()
    );
}

#[test]
fn decoder_rejects_noncanonical_maps_trailing_bytes_and_oversized_inputs() {
    let identity = stage_identity();
    let value = serde_json::to_value(&identity).unwrap();
    let object = value.as_object().unwrap();
    let mut entries: Vec<_> = object.iter().collect();
    entries.sort_unstable_by(|left, right| right.0.as_bytes().cmp(left.0.as_bytes()));

    let mut reordered = CODEC_PREFIX.to_vec();
    let mut encoder = Encoder::new(&mut reordered);
    encoder
        .array(2)
        .and_then(|encoder| encoder.str(MeshingStageIdentity::DOMAIN))
        .and_then(|encoder| encoder.map(entries.len() as u64))
        .unwrap();
    for (key, value) in entries {
        encoder.str(key).unwrap();
        runmat_canonical_codec::encode_json_value(&mut encoder, value).unwrap();
    }
    assert_eq!(
        MeshingStageIdentity::canonical_decode(&reordered)
            .unwrap_err()
            .field,
        "canonical decoding"
    );

    let mut trailing = identity.canonical_encode().unwrap();
    trailing.push(0);
    assert!(MeshingStageIdentity::canonical_decode(&trailing).is_err());

    let oversized = vec![0_u8; MeshingStageIdentity::LIMITS.maximum_encoded_bytes + 1];
    assert_eq!(
        MeshingStageIdentity::canonical_decode(&oversized)
            .unwrap_err()
            .reason,
        "encoded contract exceeds its byte limit"
    );
}

#[test]
fn decoder_rejects_collection_and_nesting_limits_before_allocation() {
    let envelope = |write_value: fn(&mut Encoder<&mut Vec<u8>>)| {
        let mut bytes = CODEC_PREFIX.to_vec();
        let mut encoder = Encoder::new(&mut bytes);
        encoder
            .array(2)
            .and_then(|encoder| encoder.str(MeshingStageIdentity::DOMAIN))
            .unwrap();
        write_value(&mut encoder);
        bytes
    };
    let oversized_collection = envelope(|encoder| {
        encoder
            .array((MeshingStageIdentity::LIMITS.maximum_collection_items + 1) as u64)
            .unwrap();
    });
    assert!(MeshingStageIdentity::canonical_decode(&oversized_collection).is_err());

    let excessive_nesting = envelope(|encoder| {
        for _ in 0..=MeshingStageIdentity::LIMITS.maximum_nesting_depth {
            encoder.array(1).unwrap();
        }
        encoder.null().unwrap();
    });
    assert!(MeshingStageIdentity::canonical_decode(&excessive_nesting).is_err());
}

#[test]
fn every_identity_manifest_and_result_contract_round_trips() {
    fn round_trip<T>(value: &T)
    where
        T: CanonicalMeshingContract + PartialEq + std::fmt::Debug,
    {
        let bytes = value.canonical_encode().unwrap();
        assert_eq!(&T::canonical_decode(&bytes).unwrap(), value);
    }

    let stage = stage_identity();
    round_trip(&stage);
    round_trip(&batch_partition());
    round_trip(&MeshingPartitionIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage_identity_digest: digest(31),
        partition: batch_partition(),
    });
    round_trip(&MeshingJoinIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage_identity_digest: digest(31),
        join_algorithm_version: "surface-stitch/v2".into(),
        ordered_partition_results: vec![
            MeshingPartitionResultRef {
                partition_index: 0,
                result_digest: digest(32),
            },
            MeshingPartitionResultRef {
                partition_index: 1,
                result_digest: digest(33),
            },
        ],
    });
    round_trip(&MeshingStageResultIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage: MeshingStageKind::SurfaceMesh,
        result_kind: MeshingStageResultKind::DeterministicJoin,
        producer_identity_digest: digest(34),
        logical_content_digest: digest(35),
        logical_entity_count: 16,
        invariant_summary_digest: digest(37),
    });
    round_trip(&MeshingValidationIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        subject_stage_result_digest: digest(38),
        geometry: stage.geometry,
        resolved_request_digest: stage.resolved_request_digest,
        validation_algorithm_version: "independent-validation/v2".into(),
        capability_cohort: stage.capability_cohort,
    });

    let manifest = MeshingStageManifest {
        schema_version: MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
        stage: MeshingStageKind::SurfaceMesh,
        result_kind: MeshingStageResultKind::DeterministicJoin,
        logical_result_identity: digest(39),
        disposition: MeshingManifestDisposition::ValidatedDependency,
        prerequisite_manifest_digests: vec![digest(40)],
        invariant_summary_digest: digest(41),
        chunks: vec![MeshingChunkDescriptor {
            ordinal: 0,
            first_logical_entity_ordinal: 0,
            digest: digest(42),
            media_type: MeshingChunkMediaType::SurfacePartitions,
            schema_version: 2,
            encoded_length: 100,
            decoded_length: 200,
            logical_entity_count: 16,
        }],
        total_encoded_length: 100,
    };
    round_trip(&manifest);
    round_trip(&MeshingWorkloadResult::Validated {
        stage_manifest_digest: manifest.canonical_digest().unwrap(),
    });
}

#[test]
fn floating_point_identity_preserves_negative_zero_bits() {
    let failure = |value| MeshingFailure {
        schema_version: MESHING_FAILURE_SCHEMA_VERSION,
        category: MeshingFailureCategory::NumericalFailure,
        stage: MeshingStageKind::Optimization,
        operation: MeshingOperation::Optimize,
        entity_ids: Vec::new(),
        witnesses: Vec::new(),
        request_values: vec![crate::contracts::canonical::MeshingDiagnosticEntry {
            name: "signed_zero".into(),
            value: MeshingDiagnosticValue::Scalar(value),
            unit: None,
        }],
        achieved_values: Vec::new(),
        remediation: "inspect the deterministic numerical witness".into(),
    };
    assert_ne!(
        failure(0.0).canonical_digest().unwrap(),
        failure(-0.0).canonical_digest().unwrap()
    );
}

#[test]
fn artifact_digest_is_sealed_over_payload_and_detects_tampering() {
    let mut artifact = artifact();
    let digest = artifact.seal_canonical_digest().unwrap();
    assert_eq!(artifact.canonical_digest().unwrap(), digest);

    let encoded = artifact.canonical_encode().unwrap();
    let decoded = SolverMeshArtifact::canonical_decode(&encoded).unwrap();
    assert_eq!(decoded, artifact);

    let mut tampered = artifact.clone();
    tampered.topology.nodes[0].coordinates_m[0] = 0.25;
    assert_eq!(
        tampered.validate_canonical().unwrap_err().field,
        "artifact.canonical_digest"
    );

    let evidence = evidence(&artifact);
    let evidence_bytes = evidence.canonical_encode().unwrap();
    let decoded_evidence =
        crate::contracts::canonical::MeshingEvidence::canonical_decode(&evidence_bytes).unwrap();
    decoded_evidence.validate(&artifact).unwrap();
}

#[test]
fn workload_and_progress_round_trip_through_the_bounded_codec() {
    let workload = workload();
    let encoded = workload.canonical_encode().unwrap();
    assert_eq!(
        crate::contracts::canonical::MeshingWorkloadRequest::canonical_decode(&encoded).unwrap(),
        workload
    );

    let progress = progress(3, 12);
    let encoded = progress.canonical_encode().unwrap();
    assert_eq!(
        MeshingProgress::canonical_decode(&encoded).unwrap(),
        progress
    );
}
