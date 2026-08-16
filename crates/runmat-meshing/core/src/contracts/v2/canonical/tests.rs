use std::collections::BTreeMap;

use minicbor::Encoder;

use super::{CanonicalMeshingContract, CODEC_PREFIX};
use crate::contracts::v2::artifact_tests::{artifact, evidence, request};
use crate::contracts::v2::execution_contract_tests::{
    batch_partition, progress, stage_identity, workload,
};
use crate::contracts::v2::{
    AnalysisMeshArtifactV2, GeometryTolerancePolicy, MeshingChunkDescriptorV2,
    MeshingChunkMediaTypeV2, MeshingDiagnosticValue, MeshingFailure, MeshingFailureCategory,
    MeshingJoinIdentityV2, MeshingManifestDispositionV2, MeshingOperationV2,
    MeshingPartitionIdentityV2, MeshingPartitionResultRefV2, MeshingProgressV2, MeshingRequestV2,
    MeshingStageIdentityV2, MeshingStageManifestV2, MeshingStageResultIdentityV2,
    MeshingStageResultKindV2, MeshingStageV2, MeshingValidationIdentityV2, MeshingWorkloadResultV2,
    StableDigest, MESHING_FAILURE_SCHEMA_VERSION, MESHING_IDENTITY_SCHEMA_VERSION,
    MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
};

fn digest(byte: u8) -> StableDigest {
    StableDigest::from_bytes([byte; 32])
}

#[test]
fn canonical_request_has_a_stable_golden_identity() {
    let request = request();
    let encoded = request.canonical_encode().unwrap();
    let decoded = MeshingRequestV2::canonical_decode(&encoded).unwrap();
    assert_eq!(decoded, request);
    assert_eq!(decoded.canonical_encode().unwrap(), encoded);
    assert_eq!(
        request.canonical_digest().unwrap().bytes(),
        &[
            103, 163, 170, 116, 66, 70, 137, 109, 83, 158, 137, 107, 201, 16, 150, 8, 168, 65, 245,
            29, 233, 42, 225, 112, 177, 162, 122, 215, 164, 92, 114, 186,
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
    let error = MeshingStageIdentityV2::canonical_decode(&encoded).unwrap_err();
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
        .and_then(|encoder| encoder.str(MeshingStageIdentityV2::DOMAIN))
        .and_then(|encoder| encoder.map(entries.len() as u64))
        .unwrap();
    for (key, value) in entries {
        encoder.str(key).unwrap();
        super::value::encode_value(&mut encoder, value).unwrap();
    }
    assert_eq!(
        MeshingStageIdentityV2::canonical_decode(&reordered)
            .unwrap_err()
            .field,
        "canonical decoding"
    );

    let mut trailing = identity.canonical_encode().unwrap();
    trailing.push(0);
    assert!(MeshingStageIdentityV2::canonical_decode(&trailing).is_err());

    let oversized = vec![0_u8; MeshingStageIdentityV2::LIMITS.maximum_encoded_bytes + 1];
    assert_eq!(
        MeshingStageIdentityV2::canonical_decode(&oversized)
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
            .and_then(|encoder| encoder.str(MeshingStageIdentityV2::DOMAIN))
            .unwrap();
        write_value(&mut encoder);
        bytes
    };
    let oversized_collection = envelope(|encoder| {
        encoder
            .array((MeshingStageIdentityV2::LIMITS.maximum_collection_items + 1) as u64)
            .unwrap();
    });
    assert!(MeshingStageIdentityV2::canonical_decode(&oversized_collection).is_err());

    let excessive_nesting = envelope(|encoder| {
        for _ in 0..=MeshingStageIdentityV2::LIMITS.maximum_nesting_depth {
            encoder.array(1).unwrap();
        }
        encoder.null().unwrap();
    });
    assert!(MeshingStageIdentityV2::canonical_decode(&excessive_nesting).is_err());
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
    round_trip(&MeshingPartitionIdentityV2 {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage_identity_digest: digest(31),
        partition: batch_partition(),
    });
    round_trip(&MeshingJoinIdentityV2 {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage_identity_digest: digest(31),
        join_algorithm_version: "surface-stitch/v2".into(),
        ordered_partition_results: vec![
            MeshingPartitionResultRefV2 {
                partition_index: 0,
                result_digest: digest(32),
            },
            MeshingPartitionResultRefV2 {
                partition_index: 1,
                result_digest: digest(33),
            },
        ],
    });
    round_trip(&MeshingStageResultIdentityV2 {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage: MeshingStageV2::SurfaceMesh,
        result_kind: MeshingStageResultKindV2::DeterministicJoin,
        producer_identity_digest: digest(34),
        logical_content_digest: digest(35),
        logical_entity_count: 16,
        invariant_summary_digest: digest(37),
    });
    round_trip(&MeshingValidationIdentityV2 {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        subject_stage_result_digest: digest(38),
        geometry: stage.geometry,
        resolved_request_digest: stage.resolved_request_digest,
        validation_algorithm_version: "independent-validation/v2".into(),
        capability_cohort: stage.capability_cohort,
    });

    let manifest = MeshingStageManifestV2 {
        schema_version: MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
        stage: MeshingStageV2::SurfaceMesh,
        result_kind: MeshingStageResultKindV2::DeterministicJoin,
        logical_result_identity: digest(39),
        disposition: MeshingManifestDispositionV2::ValidatedDependency,
        prerequisite_manifest_digests: vec![digest(40)],
        invariant_summary_digest: digest(41),
        chunks: vec![MeshingChunkDescriptorV2 {
            ordinal: 0,
            first_logical_entity_ordinal: 0,
            digest: digest(42),
            media_type: MeshingChunkMediaTypeV2::SurfacePartitions,
            schema_version: 2,
            encoded_length: 100,
            decoded_length: 200,
            logical_entity_count: 16,
        }],
        total_encoded_length: 100,
    };
    round_trip(&manifest);
    round_trip(&MeshingWorkloadResultV2::Validated {
        stage_manifest_digest: manifest.canonical_digest().unwrap(),
    });
}

#[test]
fn floating_point_identity_preserves_negative_zero_bits() {
    let failure = |value| MeshingFailure {
        schema_version: MESHING_FAILURE_SCHEMA_VERSION,
        category: MeshingFailureCategory::NumericalFailure,
        stage: MeshingStageV2::Optimization,
        operation: MeshingOperationV2::Optimize,
        entity_ids: Vec::new(),
        witnesses: Vec::new(),
        request_values: vec![crate::contracts::v2::MeshingDiagnosticEntry {
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
    let decoded = AnalysisMeshArtifactV2::canonical_decode(&encoded).unwrap();
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
        crate::contracts::v2::MeshingEvidenceV2::canonical_decode(&evidence_bytes).unwrap();
    decoded_evidence.validate(&artifact).unwrap();
}

#[test]
fn workload_and_progress_round_trip_through_the_bounded_codec() {
    let workload = workload();
    let encoded = workload.canonical_encode().unwrap();
    assert_eq!(
        crate::contracts::v2::MeshingWorkloadRequestV2::canonical_decode(&encoded).unwrap(),
        workload
    );

    let progress = progress(3, 12);
    let encoded = progress.canonical_encode().unwrap();
    assert_eq!(
        MeshingProgressV2::canonical_decode(&encoded).unwrap(),
        progress
    );
}
