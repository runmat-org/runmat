use super::*;

#[test]
fn decode_figure_bundle_rejects_hostile_counts_before_allocating() {
    let mut bytes = FIG_V1_MAGIC.to_vec();
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    let err = block_on(decode_figure_bundle(None, &bytes, OPENFIG)).expect_err("hostile count");
    assert!(err.message().contains("too many figures"));
}

#[test]
fn decode_figure_bundle_rejects_oversized_payload_length() {
    let mut bytes = FIG_V1_MAGIC.to_vec();
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(
        &(ReplayLimits::default().max_scene_payload_bytes as u64 + 1).to_le_bytes(),
    );
    let err = block_on(decode_figure_bundle(None, &bytes, OPENFIG)).expect_err("oversized payload");
    assert!(err.message().contains("exceeds maximum scene payload size"));
}

#[test]
fn archive_profiles_are_encoded_and_round_trip_without_becoming_mat_files() {
    let payloads = vec![br#"{"figure":1}"#.to_vec(), br#"{"figure":2}"#.to_vec()];
    for (profile, tag, flags) in [
        (FigureArchiveProfile::Default, 0, FLAG_COMPRESSED),
        (FigureArchiveProfile::V6, 1, 0),
        (FigureArchiveProfile::V7, 2, FLAG_COMPRESSED),
        (FigureArchiveProfile::V73, 3, FLAG_COMPRESSED),
        (FigureArchiveProfile::Compact, 4, FLAG_COMPRESSED),
    ] {
        let archive = block_on(encode_figure_bundle(None, &payloads, profile, SAVEFIG))
            .expect("encode archive");
        assert!(archive.starts_with(FIG_V2_MAGIC));
        assert_eq!(archive[FIG_V2_MAGIC.len()], tag);
        assert_eq!(archive[FIG_V2_MAGIC.len() + 1], flags);
        assert_eq!(
            block_on(decode_figure_bundle(None, &archive, OPENFIG)).expect("decode archive"),
            payloads
        );
    }
}

#[test]
fn archive_rejects_a_profile_compression_mismatch() {
    let payload = vec![br#"{"figure":1}"#.to_vec()];
    let mut archive = block_on(encode_figure_bundle(
        None,
        &payload,
        FigureArchiveProfile::V6,
        SAVEFIG,
    ))
    .expect("encode uncompressed archive");
    archive[FIG_V2_MAGIC.len() + 1] = FLAG_COMPRESSED;
    let error =
        block_on(decode_figure_bundle(None, &archive, OPENFIG)).expect_err("mismatched profile");
    assert!(error.message().contains("corrupt FIG bundle"));
}

#[test]
fn archive_rejects_unconsumed_bytes_inside_a_compressed_entry() {
    let payload = vec![br#"{"figure":1}"#.to_vec()];
    let archive = block_on(encode_figure_bundle(
        None,
        &payload,
        FigureArchiveProfile::Default,
        SAVEFIG,
    ))
    .expect("encode compressed archive");
    let header_len = FIG_V2_MAGIC.len() + 2 + 4;
    let stored_len_offset = header_len + 8;
    let stored_len = u64::from_le_bytes(
        archive[stored_len_offset..stored_len_offset + 8]
            .try_into()
            .expect("stored length"),
    );
    let payload_offset = stored_len_offset + 8;
    let mut corrupted = archive[..payload_offset + stored_len as usize].to_vec();
    corrupted.extend_from_slice(b"junk");
    corrupted[stored_len_offset..stored_len_offset + 8]
        .copy_from_slice(&(stored_len + 4).to_le_bytes());

    let error = block_on(decode_figure_bundle(None, &corrupted, OPENFIG))
        .expect_err("compressed entry must consume its complete extent");
    assert!(error.message().contains("corrupt FIG bundle"));
}

#[test]
fn archive_rejects_aggregate_decoded_extent_before_decompression() {
    let per_scene = ReplayLimits::default().max_scene_payload_bytes;
    let count = MAX_FIGURE_ARCHIVE_BYTES / per_scene + 1;
    assert!(count <= MAX_FIGURE_BUNDLE_COUNT);
    let mut archive = FIG_V2_MAGIC.to_vec();
    archive.push(0); // Default profile.
    archive.push(FLAG_COMPRESSED);
    archive.extend_from_slice(&(count as u32).to_le_bytes());
    for _ in 0..count {
        archive.extend_from_slice(&(per_scene as u64).to_le_bytes());
        archive.extend_from_slice(&0u64.to_le_bytes());
    }

    let error = block_on(decode_figure_bundle(None, &archive, OPENFIG))
        .expect_err("aggregate decoded extent");
    assert!(error.message().contains("decoded FIG archive exceeds"));
    assert!(!error.message().contains("decompress"));
}

#[test]
fn archive_cpu_paths_observe_invocation_cancellation_before_publication() {
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(runmat_execution::CancellationReason::User);
    let payloads = vec![vec![7u8; 16 * 1024]];

    let encode_error = block_on(encode_figure_bundle(
        Some(&runtime),
        &payloads,
        FigureArchiveProfile::Default,
        SAVEFIG,
    ))
    .expect_err("cancelled compression");
    assert!(encode_error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));

    let archive = block_on(encode_figure_bundle(
        None,
        &payloads,
        FigureArchiveProfile::Default,
        SAVEFIG,
    ))
    .expect("uncancelled archive fixture");
    let decode_error = block_on(decode_figure_bundle(Some(&runtime), &archive, OPENFIG))
        .expect_err("cancelled decompression");
    assert!(decode_error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
}
