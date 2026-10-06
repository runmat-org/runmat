use super::*;

pub(super) fn read_count(
    bytes: &[u8],
    cursor: &mut usize,
    builtin: &'static str,
) -> BuiltinResult<usize> {
    let count = usize::try_from(read_u32(bytes, cursor, builtin)?).map_err(|_| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG bundle count does not fit this platform"),
        )
    })?;
    if count == 0 || count > MAX_FIGURE_BUNDLE_COUNT {
        let detail = if count == 0 {
            "contains no figures"
        } else {
            "contains too many figures"
        };
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG bundle {detail}"),
        ));
    }
    Ok(count)
}

pub(super) fn read_payload_len(
    bytes: &[u8],
    cursor: &mut usize,
    builtin: &'static str,
) -> BuiltinResult<usize> {
    let len = usize::try_from(read_u64(bytes, cursor, builtin)?).map_err(|_| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG payload length does not fit this platform"),
        )
    })?;
    if len > ReplayLimits::default().max_scene_payload_bytes {
        return Err(payload_too_large(builtin, &ERROR_IMPORT));
    }
    Ok(len)
}

pub(super) fn archive_expands_too_large(builtin: &'static str) -> RuntimeError {
    persistence_error(
        &ERROR_IMPORT,
        builtin,
        format!("{builtin}: decoded FIG archive exceeds maximum size"),
    )
}

fn read_u32(bytes: &[u8], cursor: &mut usize, builtin: &'static str) -> BuiltinResult<u32> {
    let end = cursor.checked_add(4).ok_or_else(|| corrupt(builtin))?;
    let slice = bytes.get(*cursor..end).ok_or_else(|| truncated(builtin))?;
    *cursor = end;
    Ok(u32::from_le_bytes(
        slice.try_into().expect("u32 slice length"),
    ))
}

fn read_u64(bytes: &[u8], cursor: &mut usize, builtin: &'static str) -> BuiltinResult<u64> {
    let end = cursor.checked_add(8).ok_or_else(|| corrupt(builtin))?;
    let slice = bytes.get(*cursor..end).ok_or_else(|| truncated(builtin))?;
    *cursor = end;
    Ok(u64::from_le_bytes(
        slice.try_into().expect("u64 slice length"),
    ))
}
