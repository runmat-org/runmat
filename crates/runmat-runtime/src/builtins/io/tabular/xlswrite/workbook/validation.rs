//! Existing package bounds and RunMat ownership marker admission.

use super::*;

pub(super) async fn preflight_existing_workbook(
    bytes: &[u8],
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
) -> BuiltinResult<()> {
    if !looks_like_zip(bytes) {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            "xlswrite: updating existing legacy BIFF .xls workbooks is not supported",
        ));
    }
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|err| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!("xlswrite: invalid ZIP workbook: {err}"),
        )
    })?;
    if archive.len() > MAX_XLSWRITE_ZIP_ENTRIES {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!("xlswrite: ZIP workbook contains more than {MAX_XLSWRITE_ZIP_ENTRIES} entries"),
        ));
    }
    let mut total = 0u64;
    for idx in 0..archive.len() {
        let entry_size = {
            let entry = archive.by_index(idx).map_err(|err| {
                xlswrite_error_with(
                    &XLSWRITE_ERROR_IO,
                    format!("xlswrite: unable to inspect ZIP workbook entry {idx}: {err}"),
                )
            })?;
            entry.size()
        };
        if entry_size > MAX_XLSWRITE_ZIP_ENTRY_UNCOMPRESSED_BYTES {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_IO,
                format!(
                    "xlswrite: ZIP workbook entry exceeds maximum expanded size of {MAX_XLSWRITE_ZIP_ENTRY_UNCOMPRESSED_BYTES} bytes"
                ),
            ));
        }
        total = total.checked_add(entry_size).ok_or_else(|| {
            xlswrite_error_with(
                &XLSWRITE_ERROR_IO,
                "xlswrite: ZIP workbook expanded size overflows supported bounds",
            )
        })?;
        if total > MAX_XLSWRITE_ZIP_TOTAL_UNCOMPRESSED_BYTES {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_IO,
                format!(
                    "xlswrite: ZIP workbook expanded size exceeds maximum of {MAX_XLSWRITE_ZIP_TOTAL_UNCOMPRESSED_BYTES} bytes"
                ),
            ));
        }
        if (idx + 1) % 1024 == 0 {
            checkpoint_existing(cooperation, selected, 1024).await?;
        }
    }
    checkpoint_existing(cooperation, selected, archive.len() % 1024).await?;
    Ok(())
}

pub(super) async fn ensure_runmat_xlswrite_workbook(
    path: &Path,
    bytes: &[u8],
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
) -> BuiltinResult<()> {
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|err| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!("xlswrite: invalid ZIP workbook: {err}"),
        )
    })?;
    let mut marker = archive.by_name(XLSWRITE_MARKER_PART).map_err(|_| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!(
                "xlswrite: refusing to update existing workbook '{}' because it was not created by RunMat xlswrite",
                path.display()
            ),
        )
    })?;
    let mut contents = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        checkpoint_existing(cooperation, selected, 0).await?;
        let read = marker.read(&mut chunk).map_err(|err| {
            xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                format!("xlswrite: unable to read RunMat workbook marker: {err}"),
                err,
            )
        })?;
        if read == 0 {
            break;
        }
        contents.try_reserve(read).map_err(|source| {
            xlswrite_error_with_source(
                &XLSWRITE_ERROR_IO,
                "xlswrite: workbook marker allocation failed",
                source,
            )
        })?;
        contents.extend_from_slice(&chunk[..read]);
        checkpoint_existing(cooperation, selected, read).await?;
    }
    let contents = String::from_utf8(contents).map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            "xlswrite: invalid UTF-8 in RunMat workbook marker",
            source,
        )
    })?;
    if !contents.contains(XLSWRITE_MARKER_TEXT) {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_IO,
            format!(
                "xlswrite: refusing to update existing workbook '{}' because its RunMat marker is invalid",
                path.display()
            ),
        ));
    }
    Ok(())
}

fn looks_like_zip(bytes: &[u8]) -> bool {
    bytes.starts_with(b"PK\x03\x04")
        || bytes.starts_with(b"PK\x05\x06")
        || bytes.starts_with(b"PK\x07\x08")
}
