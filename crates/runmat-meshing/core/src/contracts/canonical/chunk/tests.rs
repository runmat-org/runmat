use super::*;
use crate::contracts::canonical::{
    CanonicalMeshingContract, MeshingCancellationSignal, MeshingManifestDisposition,
    MeshingStageKind, MeshingStageResultKind,
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn digest(byte: u8) -> StableDigest {
    StableDigest::from_bytes([byte; 32])
}

fn streams() -> Vec<MeshingChunkStream> {
    vec![
        MeshingChunkStream {
            media_type: MeshingChunkMediaType::MeshNodes,
            schema_version: 2,
            records: (0_u8..10).map(|value| vec![value; 80]).collect(),
        },
        MeshingChunkStream {
            media_type: MeshingChunkMediaType::MeshElements,
            schema_version: 2,
            records: (10_u8..15).map(|value| vec![value; 120]).collect(),
        },
    ]
}

fn policy(maximum_chunk_bytes: u64, maximum_records_per_chunk: u32) -> MeshingChunkPolicy {
    MeshingChunkPolicy {
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
        MeshingStageKind::Serialization,
        MeshingStageResultKind::WholeStage,
        digest(1),
        digest(2),
        vec![digest(3)],
        MeshingManifestDisposition::ValidatedDependency,
        &fine,
    )
    .unwrap();
    let (coarse_identity, coarse_manifest) = build_closed_stage_manifest(
        MeshingStageKind::Serialization,
        MeshingStageResultKind::WholeStage,
        digest(1),
        digest(2),
        vec![digest(3)],
        MeshingManifestDisposition::ValidatedDependency,
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
    assert_eq!(
        decode_stage_manifest_streams(&fine_manifest, &fine_identity, &fine.chunks).unwrap(),
        streams()
    );
    assert_eq!(
        decode_stage_manifest_streams(&coarse_manifest, &coarse_identity, &coarse.chunks).unwrap(),
        streams()
    );
}

#[test]
fn closure_rejects_corruption_truncation_and_reordering() {
    let payload = build_chunked_stage_payload(&streams(), policy(512, 2)).unwrap();
    let (identity, manifest) = build_closed_stage_manifest(
        MeshingStageKind::Serialization,
        MeshingStageResultKind::WholeStage,
        digest(1),
        digest(2),
        Vec::new(),
        MeshingManifestDisposition::ValidatedDependency,
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
fn chunk_builder_fragments_oversized_records_and_enforces_hard_total_bounds() {
    let mut unordered = streams();
    unordered.swap(0, 1);
    assert_eq!(
        build_chunked_stage_payload(&unordered, policy(512, 2))
            .unwrap_err()
            .field,
        "meshing chunk streams"
    );

    let oversized_record = vec![MeshingChunkStream {
        media_type: MeshingChunkMediaType::MeshNodes,
        schema_version: 2,
        records: vec![vec![1; 500]],
    }];
    let fragmented = build_chunked_stage_payload(&oversized_record, policy(512, 2)).unwrap();
    assert!(fragmented.chunks.len() > 1);
    assert_eq!(fragmented.logical_entity_count, 1);
    assert_eq!(fragmented.chunks[0].descriptor.logical_entity_count, 0);
    let (identity, manifest) = build_closed_stage_manifest(
        MeshingStageKind::Serialization,
        MeshingStageResultKind::WholeStage,
        digest(4),
        digest(5),
        Vec::new(),
        MeshingManifestDisposition::ValidatedDependency,
        &fragmented,
    )
    .unwrap();
    assert_eq!(
        decode_stage_manifest_streams(&manifest, &identity, &fragmented.chunks).unwrap(),
        oversized_record
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
        MeshingStageKind::Serialization,
        MeshingStageResultKind::WholeStage,
        digest(1),
        digest(2),
        Vec::new(),
        MeshingManifestDisposition::ValidatedDependency,
        &payload,
    )
    .unwrap();
    manifest.chunks[1].digest = manifest.chunks[0].digest;
    assert_eq!(
        manifest.validate().unwrap_err().field,
        "meshing stage manifest chunks"
    );

    let (_, mut manifest) = build_closed_stage_manifest(
        MeshingStageKind::Serialization,
        MeshingStageResultKind::WholeStage,
        digest(1),
        digest(2),
        Vec::new(),
        MeshingManifestDisposition::ValidatedDependency,
        &payload,
    )
    .unwrap();
    manifest.chunks[1].first_logical_entity_ordinal += 1;
    assert_eq!(
        manifest.validate().unwrap_err().field,
        "meshing stage manifest chunks"
    );
}

#[test]
fn stream_reconstruction_observes_cancellation_during_chunk_processing() {
    struct CancelAfter {
        remaining: AtomicUsize,
    }

    impl MeshingCancellationSignal for CancelAfter {
        fn is_cancelled(&self) -> bool {
            self.remaining
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_err()
        }
    }

    let payload = build_chunked_stage_payload(&streams(), policy(512, 2)).unwrap();
    let (identity, manifest) = build_closed_stage_manifest(
        MeshingStageKind::Serialization,
        MeshingStageResultKind::WholeStage,
        digest(1),
        digest(2),
        Vec::new(),
        MeshingManifestDisposition::ValidatedDependency,
        &payload,
    )
    .unwrap();
    let cancellation = CancelAfter {
        remaining: AtomicUsize::new(6),
    };

    assert_eq!(
        decode_stage_manifest_streams_with_cancellation(
            &manifest,
            &identity,
            &payload.chunks,
            &cancellation,
        )
        .unwrap_err()
        .field,
        "meshing chunk processing"
    );
}
