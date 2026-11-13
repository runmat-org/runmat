use super::*;
use crate::contracts::canonical::{
    CanonicalMeshingContract, MeshingCancellationSignal, MeshingManifestDisposition,
    MeshingStageKind, MeshingStageResultKind, NeverCancelled,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn preflight_matches_actual_capacity_across_record_and_ordinal_widths() {
    for chunk_bytes in [512, 1024, 8192] {
        let original = vec![MeshingChunkStream {
            media_type: MeshingChunkMediaType::MeshNodes,
            schema_version: 256,
            records: [1, 23, 24, 255, 256, 16000]
                .into_iter()
                .cycle()
                .take(90)
                .map(|size| vec![17; size])
                .collect(),
        }];
        let mut policy = policy(chunk_bytes, 3);
        policy.maximum_total_encoded_bytes = 1024 * 1024;
        let profile = profile_chunked_stage_payload(&original, policy, &NeverCancelled).unwrap();
        let payload = build_chunked_stage_payload(&original, policy).unwrap();
        assert_eq!(profile.chunk_count, payload.chunks.len());
        assert_eq!(profile.total_encoded_length, payload.total_encoded_length);
        let requested = payload.chunks.capacity() * std::mem::size_of::<EncodedMeshingChunk>()
            + payload
                .chunks
                .iter()
                .map(|chunk| chunk.bytes.capacity())
                .sum::<usize>();
        assert_eq!(profile.requested_storage_bytes, requested as u64);
        for chunk in &payload.chunks {
            let decoded =
                codec::decode_chunk_with_cancellation(chunk.into(), &NeverCancelled).unwrap();
            // Independent writer oracle for the unchanged v3 envelope and fields.
            let mut expected = b"runmat-meshing-logical-record-chunk/v3\0".to_vec();
            let mut encoder = minicbor::Encoder::new(&mut expected);
            encoder
                .array(6)
                .unwrap()
                .u16(3)
                .unwrap()
                .str(chunk.descriptor.media_type.media_type())
                .unwrap()
                .u16(chunk.descriptor.schema_version)
                .unwrap()
                .u32(chunk.descriptor.ordinal)
                .unwrap()
                .u64(chunk.descriptor.first_logical_entity_ordinal)
                .unwrap()
                .array(decoded.fragments.len() as u64)
                .unwrap();
            for fragment in decoded.fragments.iter() {
                encoder
                    .array(2)
                    .unwrap()
                    .bool(fragment.completes_record)
                    .unwrap()
                    .bytes(fragment.bytes)
                    .unwrap();
            }
            assert_eq!(chunk.bytes, expected);
        }
    }
}

#[test]
fn chunk_preflight_rejects_total_limit_and_observes_cancellation() {
    let mut limited = policy(512, 2);
    limited.maximum_total_encoded_bytes = 512;
    assert!(profile_chunked_stage_payload(&streams(), limited, &NeverCancelled).is_err());
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    assert_eq!(
        profile_chunked_stage_payload(&streams(), policy(512, 2), &Cancelled)
            .unwrap_err()
            .field,
        "meshing chunk processing"
    );
}

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

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Fragmented records retain the original logical digest with borrowed payload fragments
//   and reject forged content. This component check does not establish native IO throughput or
//   whole-task memory admission.
#[test]
fn streaming_closure_matches_original_content_across_fragmentation_and_rejects_forged_digest() {
    for size in [512, 1024, 8192] {
        let original = vec![MeshingChunkStream {
            media_type: MeshingChunkMediaType::MeshNodes,
            schema_version: 1,
            records: vec![vec![7; 16384], vec![9; 71], vec![3; 32768]],
        }];
        let payload = build_chunked_stage_payload(&original, policy(size, 2)).unwrap();
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
        let borrowed = payload.chunks.iter().map(Into::into).collect::<Vec<_>>();
        verify_stage_manifest_closure_borrowed(
            &manifest,
            &identity,
            &borrowed,
            &crate::NeverCancelled,
        )
        .unwrap();
        assert_eq!(
            decode_stage_manifest_streams(&manifest, &identity, &payload.chunks).unwrap(),
            original
        );
        let mut forged = identity.clone();
        forged.logical_content_digest = digest(99);
        let mut forged_manifest = manifest.clone();
        forged_manifest.logical_result_identity = forged.canonical_digest().unwrap();
        assert!(verify_stage_manifest_closure_borrowed(
            &forged_manifest,
            &forged,
            &borrowed,
            &crate::NeverCancelled
        )
        .is_err());
        let decoded =
            codec::decode_chunk_with_cancellation(borrowed[0], &crate::NeverCancelled).unwrap();
        let begin = payload.chunks[0].bytes.as_ptr() as usize;
        let end = begin + payload.chunks[0].bytes.len();
        assert!(decoded.fragments.iter().all(|fragment| {
            let address = fragment.bytes.as_ptr() as usize;
            address >= begin && address + fragment.bytes.len() <= end
        }));
    }
}
