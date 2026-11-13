use super::*;
use crate::{build_chunked_stage_payload, MeshingChunkMediaType, MeshingChunkPolicy};
fn payload() -> MeshingChunkedPayload {
    build_chunked_stage_payload(
        &[MeshingChunkStream {
            media_type: MeshingChunkMediaType::MeshNodes,
            schema_version: 2,
            records: vec![vec![1; 13000], vec![2; 100]],
        }],
        MeshingChunkPolicy {
            maximum_chunk_bytes: 512,
            maximum_records_per_chunk: 10,
            maximum_total_encoded_bytes: 1 << 20,
        },
    )
    .unwrap()
}
fn metadata() -> MeshingManifestMetadata {
    MeshingManifestMetadata {
        stage: MeshingStageKind::Tetrahedralization,
        result_kind: MeshingStageResultKind::WholeStage,
        producer_identity_digest: StableDigest::from_bytes([1; 32]),
        invariant_summary_digest: StableDigest::from_bytes([2; 32]),
        disposition: MeshingManifestDisposition::ValidatedDependency,
    }
}
#[test]
fn closing_moves_original_chunks_and_preserves_raw_independent_verification() {
    let payload = payload();
    let pointers = payload
        .chunks
        .iter()
        .map(|chunk| chunk.bytes.as_ptr())
        .collect::<Vec<_>>();
    let mut calls = 0;
    let closed =
        close_stage_payload_with_admission(metadata(), vec![], payload, &NeverCancelled, |_| {
            calls += 1;
            Ok(())
        })
        .unwrap();
    let (identity, manifest, mut chunks) = closed.into_parts();
    for (chunk, original) in chunks.iter().zip(pointers) {
        assert_eq!(chunk.bytes.as_ptr(), original);
    }
    verify_stage_manifest_closure(&manifest, &identity, &chunks).unwrap();
    chunks[0].bytes[0] ^= 1;
    assert!(verify_stage_manifest_closure(&manifest, &identity, &chunks).is_err());
    for denied in 0..calls {
        let mut observed = 0;
        let result = close_stage_payload_with_admission(
            metadata(),
            vec![],
            self::payload(),
            &NeverCancelled,
            |_| {
                let ordinal = observed;
                observed += 1;
                if ordinal == denied {
                    Err(invalid("injected metadata denial"))
                } else {
                    Ok(())
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(observed, denied + 1);
    }
}

#[test]
fn hostile_logical_counts_reject_original_chunk_grammar_before_length_allocation() {
    let payload = payload();
    let (mut identity, mut manifest) = build_closed_stage_manifest(
        MeshingStageKind::Tetrahedralization,
        MeshingStageResultKind::WholeStage,
        StableDigest::from_bytes([1; 32]),
        StableDigest::from_bytes([2; 32]),
        vec![],
        MeshingManifestDisposition::ValidatedDependency,
        &payload,
    )
    .unwrap();
    // One bounded chunk view, with a forged count whose length array would be enormous.
    let mut chunks = [payload.chunks.into_iter().next().unwrap()];
    chunks[0].descriptor.logical_entity_count = 1 << 40;
    chunks[0].descriptor.first_logical_entity_ordinal = 0;
    identity.logical_entity_count = 1 << 40;
    manifest.chunks = vec![chunks[0].descriptor.clone()];
    manifest.total_encoded_length = chunks[0].descriptor.encoded_length;
    manifest.logical_result_identity = identity.canonical_digest().unwrap();
    let borrowed = chunks.iter().map(Into::into).collect::<Vec<_>>();
    assert!(verify_stage_manifest_closure_borrowed(
        &manifest,
        &identity,
        &borrowed,
        &NeverCancelled
    )
    .is_err());
    assert!(decode_stage_manifest_streams_borrowed(
        &manifest,
        &identity,
        &borrowed,
        &NeverCancelled
    )
    .is_err());
}
