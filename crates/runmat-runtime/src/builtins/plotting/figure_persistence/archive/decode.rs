use super::*;
use crate::execution::staged_file::{StagedCpuCooperation, StagedWriteCooperation};

mod header;
mod inflate;
use header::*;
use inflate::decompress_payload;

pub(in super::super) async fn decode_figure_bundle(
    runtime: Option<&crate::context::RuntimeContext>,
    bytes: &[u8],
    builtin: &'static str,
) -> BuiltinResult<Vec<Vec<u8>>> {
    if bytes.len() > MAX_FIGURE_ARCHIVE_BYTES {
        return Err(archive_expands_too_large(builtin));
    }
    if bytes.starts_with(FIG_V2_MAGIC) {
        return decode_v2(runtime, bytes, builtin).await;
    }
    if bytes.starts_with(FIG_V1_MAGIC) {
        return decode_v1(runtime, bytes, builtin).await;
    }
    if bytes.len() > ReplayLimits::default().max_scene_payload_bytes {
        return Err(payload_too_large(builtin, &ERROR_IMPORT));
    }
    let mut cooperation = StagedCpuCooperation::new(runtime, builtin);
    Ok(vec![copy_payload(bytes, &mut cooperation).await?])
}

async fn decode_v2(
    runtime: Option<&crate::context::RuntimeContext>,
    bytes: &[u8],
    builtin: &'static str,
) -> BuiltinResult<Vec<Vec<u8>>> {
    let mut cursor = FIG_V2_MAGIC.len();
    let profile = profile_from_tag(
        *bytes.get(cursor).ok_or_else(|| truncated(builtin))?,
        builtin,
    )?;
    cursor += 1;
    let flags = *bytes.get(cursor).ok_or_else(|| truncated(builtin))?;
    cursor += 1;
    if flags & !FLAG_COMPRESSED != 0 {
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG archive contains unsupported flags"),
        ));
    }
    let compressed = matches!(
        profile,
        FigureArchiveProfile::Default
            | FigureArchiveProfile::V7
            | FigureArchiveProfile::V73
            | FigureArchiveProfile::Compact
    );
    if (flags & FLAG_COMPRESSED != 0) != compressed {
        return Err(corrupt(builtin));
    }
    let count = read_count(bytes, &mut cursor, builtin)?;
    let mut descriptors = Vec::with_capacity(count);
    let mut total_decoded = 0usize;
    for _ in 0..count {
        let decoded_len = read_payload_len(bytes, &mut cursor, builtin)?;
        total_decoded = total_decoded
            .checked_add(decoded_len)
            .filter(|total| *total <= MAX_FIGURE_ARCHIVE_BYTES)
            .ok_or_else(|| archive_expands_too_large(builtin))?;
        let stored_len = read_payload_len(bytes, &mut cursor, builtin)?;
        let end = cursor
            .checked_add(stored_len)
            .ok_or_else(|| corrupt(builtin))?;
        bytes.get(cursor..end).ok_or_else(|| truncated(builtin))?;
        if !compressed && stored_len != decoded_len {
            return Err(corrupt(builtin));
        }
        descriptors.push((decoded_len, cursor, end));
        cursor = end;
    }
    ensure_consumed(bytes, cursor, builtin)?;

    let mut cooperation = StagedCpuCooperation::new(runtime, builtin);
    let mut payloads = Vec::with_capacity(count);
    for (decoded_len, start, end) in descriptors {
        let stored = &bytes[start..end];
        payloads.push(if compressed {
            decompress_payload(stored, decoded_len, builtin, &mut cooperation).await?
        } else {
            copy_payload(stored, &mut cooperation).await?
        });
    }
    Ok(payloads)
}

async fn decode_v1(
    runtime: Option<&crate::context::RuntimeContext>,
    bytes: &[u8],
    builtin: &'static str,
) -> BuiltinResult<Vec<Vec<u8>>> {
    let mut cursor = FIG_V1_MAGIC.len();
    let count = read_count(bytes, &mut cursor, builtin)?;
    let mut descriptors = Vec::with_capacity(count);
    for _ in 0..count {
        let len = read_payload_len(bytes, &mut cursor, builtin)?;
        let end = cursor.checked_add(len).ok_or_else(|| corrupt(builtin))?;
        bytes.get(cursor..end).ok_or_else(|| truncated(builtin))?;
        descriptors.push((cursor, end));
        cursor = end;
    }
    ensure_consumed(bytes, cursor, builtin)?;
    let mut cooperation = StagedCpuCooperation::new(runtime, builtin);
    let mut payloads = Vec::with_capacity(count);
    for (start, end) in descriptors {
        payloads.push(copy_payload(&bytes[start..end], &mut cooperation).await?);
    }
    Ok(payloads)
}

async fn copy_payload(
    stored: &[u8],
    cooperation: &mut StagedCpuCooperation<'_>,
) -> BuiltinResult<Vec<u8>> {
    let mut payload = Vec::with_capacity(stored.len());
    for chunk in stored.chunks(8 * 1024) {
        cooperation.completed(chunk.len()).await?;
        payload.extend_from_slice(chunk);
    }
    Ok(payload)
}

fn ensure_consumed(bytes: &[u8], cursor: usize, builtin: &'static str) -> BuiltinResult<()> {
    if cursor != bytes.len() {
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG bundle has trailing bytes"),
        ));
    }
    Ok(())
}
