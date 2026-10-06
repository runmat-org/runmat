use super::*;

pub(super) fn encode_figure_bundle(
    payloads: &[Vec<u8>],
    builtin: &'static str,
) -> BuiltinResult<Vec<u8>> {
    if payloads.len() == 1 {
        return Ok(payloads[0].clone());
    }
    let count = u32::try_from(payloads.len()).map_err(|_| {
        persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: too many figures to save in one FIG file"),
        )
    })?;
    let mut out = Vec::new();
    out.extend_from_slice(FIG_MAGIC);
    out.extend_from_slice(&count.to_le_bytes());
    for payload in payloads {
        let len = u64::try_from(payload.len()).map_err(|_| {
            persistence_error(
                &ERROR_INVALID_ARGUMENT,
                builtin,
                format!("{builtin}: figure payload is too large"),
            )
        })?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(payload);
    }
    Ok(out)
}

pub(super) fn decode_figure_bundle<'a>(
    bytes: &'a [u8],
    builtin: &'static str,
) -> BuiltinResult<Vec<&'a [u8]>> {
    if !bytes.starts_with(FIG_MAGIC) {
        return Ok(vec![bytes]);
    }
    let mut cursor = FIG_MAGIC.len();
    let count = usize::try_from(read_u32(bytes, &mut cursor, builtin)?).map_err(|_| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG bundle count does not fit this platform"),
        )
    })?;
    if count == 0 {
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG bundle contains no figures"),
        ));
    }
    if count > MAX_FIGURE_BUNDLE_COUNT {
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG bundle contains too many figures"),
        ));
    }
    let remaining = bytes.len().saturating_sub(cursor);
    if count > remaining / 8 {
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: truncated FIG bundle"),
        ));
    }
    let mut payloads = Vec::with_capacity(count);
    for _ in 0..count {
        let len = usize::try_from(read_u64(bytes, &mut cursor, builtin)?).map_err(|_| {
            persistence_error(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: FIG payload length does not fit this platform"),
            )
        })?;
        if len > ReplayLimits::default().max_scene_payload_bytes {
            return Err(persistence_error(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: FIG payload exceeds maximum scene payload size"),
            ));
        }
        let end = cursor.checked_add(len).ok_or_else(|| {
            persistence_error(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: FIG bundle size overflow"),
            )
        })?;
        if end > bytes.len() {
            return Err(persistence_error(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: truncated FIG bundle"),
            ));
        }
        payloads.push(&bytes[cursor..end]);
        cursor = end;
    }
    if cursor != bytes.len() {
        return Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG bundle has trailing bytes"),
        ));
    }
    Ok(payloads)
}

pub(super) fn read_u32(
    bytes: &[u8],
    cursor: &mut usize,
    builtin: &'static str,
) -> BuiltinResult<u32> {
    let end = cursor.checked_add(4).ok_or_else(|| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: corrupt FIG bundle"),
        )
    })?;
    let slice = bytes.get(*cursor..end).ok_or_else(|| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: truncated FIG bundle"),
        )
    })?;
    *cursor = end;
    Ok(u32::from_le_bytes(
        slice.try_into().expect("u32 slice length"),
    ))
}

pub(super) fn read_u64(
    bytes: &[u8],
    cursor: &mut usize,
    builtin: &'static str,
) -> BuiltinResult<u64> {
    let end = cursor.checked_add(8).ok_or_else(|| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: corrupt FIG bundle"),
        )
    })?;
    let slice = bytes.get(*cursor..end).ok_or_else(|| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: truncated FIG bundle"),
        )
    })?;
    *cursor = end;
    Ok(u64::from_le_bytes(
        slice.try_into().expect("u64 slice length"),
    ))
}
