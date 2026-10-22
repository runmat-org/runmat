use super::*;
use crate::{build_chunked_stage_payload, MeshingChunkMediaType, MeshingChunkPolicy};

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Across the tested chunk sizes and fragmented multi-stream records, preflight equals
//   the actual output buffer capacities, admission denial stops reconstruction, and altered content
//   identity rejects. This finite record-reconstruction check excludes imported typed metadata,
//   constructor output, publication metadata, allocator overhead and whole-task memory
//   qualification.
#[test]
fn fragmented_reconstruction_reserves_exact_output_and_preserves_closure() {
    for chunk_bytes in [512, 4096, 65536] {
        let mut original = vec![
            MeshingChunkStream {
                media_type: MeshingChunkMediaType::MeshNodes,
                schema_version: 1,
                records: [1, 23, 24, 255, 256, 130000]
                    .map(|size| vec![17; size])
                    .to_vec(),
            },
            MeshingChunkStream {
                media_type: MeshingChunkMediaType::VolumeTopology,
                schema_version: 4,
                records: vec![vec![31; 9001], vec![19; 2]],
            },
        ];
        original.sort_by_key(|stream| stream.media_type);
        let payload = build_chunked_stage_payload(
            &original,
            MeshingChunkPolicy {
                maximum_chunk_bytes: chunk_bytes,
                maximum_records_per_chunk: 3,
                maximum_total_encoded_bytes: 1024 * 1024,
            },
        )
        .unwrap();
        let (identity, manifest) = build_closed_stage_manifest(
            MeshingStageKind::Tetrahedralization,
            MeshingStageResultKind::WholeStage,
            StableDigest::from_bytes([1; 32]),
            StableDigest::from_bytes([2; 32]),
            vec![],
            MeshingManifestDisposition::ValidatedDependency,
            &payload,
        )
        .unwrap();
        let chunks = payload.chunks.iter().map(Into::into).collect::<Vec<_>>();
        let mut admitted = None;
        let decoded = decode_stage_manifest_streams_borrowed_with_admission(
            &manifest,
            &identity,
            &chunks,
            &NeverCancelled,
            |profile| {
                assert!(admitted.replace(profile).is_none());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(decoded, original);
        let actual = decoded.capacity() * std::mem::size_of::<MeshingChunkStream>()
            + decoded
                .iter()
                .map(|stream| {
                    stream.records.capacity() * std::mem::size_of::<Vec<u8>>()
                        + stream.records.iter().map(Vec::capacity).sum::<usize>()
                })
                .sum::<usize>();
        let profile = admitted.unwrap();
        assert_eq!(profile.retained_bytes, actual as u64);
        assert_eq!(profile.scratch_bytes, {
            let layout = layout::prepare(&manifest, &identity, &chunks, &NeverCancelled).unwrap();
            (layout.lengths.capacity() * std::mem::size_of::<usize>()
                + layout.chunks.capacity() * std::mem::size_of::<codec::DecodedMeshingChunk<'_>>())
                as u64
                + MeshingStageManifest::validation_scratch_bytes(chunks.len()).unwrap()
                + identity
                    .canonical_encoding_profile()
                    .unwrap()
                    .resident_workspace_bytes_estimate
        });
        assert!(
            decode_stage_manifest_streams_borrowed_with_admission(
                &manifest,
                &identity,
                &chunks,
                &NeverCancelled,
                |_| Err(MeshingContractError::invalid(
                    "test allocation owner",
                    "denied"
                )),
            )
            .unwrap_err()
            .field
                == "test allocation owner"
        );
        let mut altered = identity.clone();
        altered.logical_content_digest = StableDigest::from_bytes([9; 32]);
        let mut altered_manifest = manifest.clone();
        altered_manifest.logical_result_identity = altered.canonical_digest().unwrap();
        assert!(decode_stage_manifest_streams_borrowed_with_admission(
            &altered_manifest,
            &altered,
            &chunks,
            &NeverCancelled,
            |_| Ok(()),
        )
        .is_err());
    }
}
