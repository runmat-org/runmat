use super::*;
use crate::contracts::v2::{
    CanonicalMeshingContract, MeshingManifestDispositionV2, MeshingStageResultKindV2,
    MeshingStageV2,
};

fn digest(byte: u8) -> StableDigest {
    StableDigest::from_bytes([byte; 32])
}

fn streams() -> Vec<MeshingChunkStreamV2> {
    vec![
        MeshingChunkStreamV2 {
            media_type: MeshingChunkMediaTypeV2::MeshNodes,
            schema_version: 2,
            records: (0_u8..10).map(|value| vec![value; 80]).collect(),
        },
        MeshingChunkStreamV2 {
            media_type: MeshingChunkMediaTypeV2::MeshElements,
            schema_version: 2,
            records: (10_u8..15).map(|value| vec![value; 120]).collect(),
        },
    ]
}

fn policy(maximum_chunk_bytes: u64, maximum_records_per_chunk: u32) -> MeshingChunkPolicyV2 {
    MeshingChunkPolicyV2 {
        maximum_chunk_bytes,
        maximum_records_per_chunk,
        maximum_total_encoded_bytes: 64 * 1024,
    }
}

#[test]
fn logical_identity_is_independent_of_deterministic_chunk_policy() {
    let fine = build_chunked_stage_payload(&streams(), policy(512, 2)).unwrap();
    let coarse = build_chunked_stage_payload(&streams(), policy(1024, 8)).unwrap();
    assert_ne!(fine.chunks.len(), coarse.chunks.len());
    assert_eq!(fine.logical_entity_count, 15);
    assert_eq!(fine.logical_content_digest, coarse.logical_content_digest);

    let (fine_identity, fine_manifest) = build_closed_stage_manifest(
        MeshingStageV2::Serialization,
        MeshingStageResultKindV2::WholeStage,
        digest(1),
        digest(2),
        vec![digest(3)],
        MeshingManifestDispositionV2::ValidatedDependency,
        &fine,
    )
    .unwrap();
    let (coarse_identity, coarse_manifest) = build_closed_stage_manifest(
        MeshingStageV2::Serialization,
        MeshingStageResultKindV2::WholeStage,
        digest(1),
        digest(2),
        vec![digest(3)],
        MeshingManifestDispositionV2::ValidatedDependency,
        &coarse,
    )
    .unwrap();
    assert_eq!(
        fine_identity.canonical_digest().unwrap(),
        coarse_identity.canonical_digest().unwrap()
    );
    assert_ne!(
        fine_manifest.canonical_digest().unwrap(),
        coarse_manifest.canonical_digest().unwrap()
    );
    verify_stage_manifest_closure(&fine_manifest, &fine_identity, &fine.chunks).unwrap();
    verify_stage_manifest_closure(&coarse_manifest, &coarse_identity, &coarse.chunks).unwrap();
}

#[test]
fn closure_rejects_corruption_truncation_and_reordering() {
    let payload = build_chunked_stage_payload(&streams(), policy(512, 2)).unwrap();
    let (identity, manifest) = build_closed_stage_manifest(
        MeshingStageV2::Serialization,
        MeshingStageResultKindV2::WholeStage,
        digest(1),
        digest(2),
        Vec::new(),
        MeshingManifestDispositionV2::ValidatedDependency,
        &payload,
    )
    .unwrap();

    let mut corrupt = payload.chunks.clone();
    *corrupt[0].bytes.last_mut().unwrap() ^= 1;
    assert_eq!(
        verify_stage_manifest_closure(&manifest, &identity, &corrupt)
            .unwrap_err()
            .field,
        "meshing chunk closure"
    );

    let mut truncated = payload.chunks.clone();
    truncated[0].bytes.pop();
    assert!(verify_stage_manifest_closure(&manifest, &identity, &truncated).is_err());

    let mut reordered = payload.chunks.clone();
    reordered.swap(0, 1);
    assert_eq!(
        verify_stage_manifest_closure(&manifest, &identity, &reordered)
            .unwrap_err()
            .field,
        "meshing stage manifest closure"
    );
}

#[test]
fn chunk_builder_enforces_stream_and_hard_byte_bounds() {
    let mut unordered = streams();
    unordered.swap(0, 1);
    assert_eq!(
        build_chunked_stage_payload(&unordered, policy(512, 2))
            .unwrap_err()
            .field,
        "meshing chunk streams"
    );

    let oversized_record = vec![MeshingChunkStreamV2 {
        media_type: MeshingChunkMediaTypeV2::MeshNodes,
        schema_version: 2,
        records: vec![vec![1; 500]],
    }];
    assert_eq!(
        build_chunked_stage_payload(&oversized_record, policy(512, 2))
            .unwrap_err()
            .field,
        "meshing chunk record"
    );

    let mut total_limited = policy(512, 2);
    total_limited.maximum_total_encoded_bytes = 512;
    assert_eq!(
        build_chunked_stage_payload(&streams(), total_limited)
            .unwrap_err()
            .field,
        "meshing chunks"
    );
}

#[test]
fn manifest_rejects_duplicate_or_discontinuous_chunk_descriptors() {
    let payload = build_chunked_stage_payload(&streams(), policy(512, 2)).unwrap();
    let (_, mut manifest) = build_closed_stage_manifest(
        MeshingStageV2::Serialization,
        MeshingStageResultKindV2::WholeStage,
        digest(1),
        digest(2),
        Vec::new(),
        MeshingManifestDispositionV2::ValidatedDependency,
        &payload,
    )
    .unwrap();
    manifest.chunks[1].digest = manifest.chunks[0].digest;
    assert_eq!(
        manifest.validate().unwrap_err().field,
        "meshing stage manifest chunks"
    );

    let (_, mut manifest) = build_closed_stage_manifest(
        MeshingStageV2::Serialization,
        MeshingStageResultKindV2::WholeStage,
        digest(1),
        digest(2),
        Vec::new(),
        MeshingManifestDispositionV2::ValidatedDependency,
        &payload,
    )
    .unwrap();
    manifest.chunks[1].first_logical_entity_ordinal += 1;
    assert_eq!(
        manifest.validate().unwrap_err().field,
        "meshing stage manifest chunks"
    );
}
