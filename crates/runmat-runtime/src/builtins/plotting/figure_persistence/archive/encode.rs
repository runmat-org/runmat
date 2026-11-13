use super::*;
use crate::execution::staged_file::StagedCpuCooperation;
use flate2::{write::ZlibEncoder, Compression};
use std::io::Write;

pub(in super::super) async fn encode_figure_bundle(
    runtime: Option<&crate::context::RuntimeContext>,
    payloads: &[Vec<u8>],
    profile: FigureArchiveProfile,
    builtin: &'static str,
) -> BuiltinResult<Vec<u8>> {
    if payloads.is_empty() {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: at least one figure is required"),
        ));
    }
    let count = u32::try_from(payloads.len()).map_err(|_| too_many_figures(builtin))?;
    if payloads.len() > MAX_FIGURE_BUNDLE_COUNT {
        return Err(too_many_figures(builtin));
    }
    let compressed = matches!(
        profile,
        FigureArchiveProfile::Default
            | FigureArchiveProfile::V7
            | FigureArchiveProfile::V73
            | FigureArchiveProfile::Compact
    );
    let mut cooperation = StagedCpuCooperation::new(runtime, builtin);
    let mut out = Vec::new();
    out.extend_from_slice(FIG_V2_MAGIC);
    out.push(profile_tag(profile));
    out.push(if compressed { FLAG_COMPRESSED } else { 0 });
    out.extend_from_slice(&count.to_le_bytes());
    let mut total_decoded = 0usize;
    for payload in payloads {
        if payload.len() > ReplayLimits::default().max_scene_payload_bytes {
            return Err(payload_too_large(builtin, &ERROR_INVALID_ARGUMENT));
        }
        total_decoded = total_decoded
            .checked_add(payload.len())
            .filter(|total| *total <= MAX_FIGURE_ARCHIVE_BYTES)
            .ok_or_else(|| archive_too_large(builtin))?;
        let stored = if compressed {
            compress_payload(payload, builtin, &mut cooperation).await?
        } else {
            copy_payload(payload, &mut cooperation).await?
        };
        out.len()
            .checked_add(16)
            .and_then(|len| len.checked_add(stored.len()))
            .filter(|len| *len <= MAX_FIGURE_ARCHIVE_BYTES)
            .ok_or_else(|| archive_too_large(builtin))?;
        push_len(&mut out, payload.len(), builtin)?;
        push_len(&mut out, stored.len(), builtin)?;
        out.extend_from_slice(&stored);
    }
    Ok(out)
}

async fn compress_payload(
    payload: &[u8],
    builtin: &'static str,
    cooperation: &mut StagedCpuCooperation<'_>,
) -> BuiltinResult<Vec<u8>> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    for chunk in payload.chunks(8 * 1024) {
        cooperation.completed(chunk.len()).await?;
        encoder.write_all(chunk).map_err(|source| {
            persistence_error_with_source(
                &ERROR_INTERNAL,
                builtin,
                format!("{builtin}: failed to compress figure payload"),
                source,
            )
        })?;
    }
    encoder.finish().map_err(|source| {
        persistence_error_with_source(
            &ERROR_INTERNAL,
            builtin,
            format!("{builtin}: failed to finish figure compression"),
            source,
        )
    })
}

async fn copy_payload(
    payload: &[u8],
    cooperation: &mut StagedCpuCooperation<'_>,
) -> BuiltinResult<Vec<u8>> {
    let mut copy = Vec::with_capacity(payload.len());
    for chunk in payload.chunks(8 * 1024) {
        cooperation.completed(chunk.len()).await?;
        copy.extend_from_slice(chunk);
    }
    Ok(copy)
}

fn push_len(out: &mut Vec<u8>, len: usize, builtin: &'static str) -> BuiltinResult<()> {
    let len = u64::try_from(len).map_err(|_| {
        persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: figure payload is too large"),
        )
    })?;
    out.extend_from_slice(&len.to_le_bytes());
    Ok(())
}

fn archive_too_large(builtin: &'static str) -> RuntimeError {
    persistence_error(
        &ERROR_INVALID_ARGUMENT,
        builtin,
        format!("{builtin}: figure archive exceeds maximum size"),
    )
}

fn too_many_figures(builtin: &'static str) -> RuntimeError {
    persistence_error(
        &ERROR_INVALID_ARGUMENT,
        builtin,
        format!("{builtin}: too many figures to save in one FIG file"),
    )
}
