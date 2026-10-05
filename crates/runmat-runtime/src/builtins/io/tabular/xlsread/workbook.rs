use std::io::{Cursor, Read};

use calamine::{open_workbook_auto_from_rs, Reader as SpreadsheetReader};

use super::super::path::TabularFsPath;
use super::range::selected_cells;
use super::*;

const MAX_XLSREAD_WORKBOOK_BYTES: u64 = 512 * 1024 * 1024;
pub(super) const MAX_XLSREAD_ZIP_ENTRY_UNCOMPRESSED_BYTES: u64 = 512 * 1024 * 1024;
const MAX_XLSREAD_ZIP_TOTAL_UNCOMPRESSED_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub(super) const MAX_XLSREAD_ZIP_ENTRIES: usize = 65_536;

pub(super) async fn read_spreadsheet(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
    request: &XlsReadRequest,
    outputs: XlsReadOutputs,
) -> BuiltinResult<XlsReadResult> {
    let bytes = read_file_bytes(runtime, cooperation, path).await?;
    preflight_zip_workbook(&bytes)?;
    let cursor = Cursor::new(bytes);
    let mut workbook = open_workbook_auto_from_rs(cursor).map_err(|err| {
        xlsread_error_with_source(
            &XLSREAD_ERROR_IO,
            format!(
                "xlsread: unable to open spreadsheet '{}': {err}",
                path.path().display()
            ),
            err,
        )
    })?;
    let range = match &request.sheet {
        Some(SheetSelector::Name(name)) => workbook.worksheet_range(name).map_err(|err| {
            xlsread_error_with_source(
                &XLSREAD_ERROR_SHEET,
                format!("xlsread: unable to read sheet '{name}': {err:?}"),
                err,
            )
        })?,
        Some(SheetSelector::Index(index)) => workbook
            .worksheet_range_at(*index)
            .ok_or_else(|| {
                xlsread_error_with(
                    &XLSREAD_ERROR_SHEET,
                    format!("xlsread: sheet index {} exceeds bounds", index + 1),
                )
            })?
            .map_err(|err| {
                xlsread_error_with_source(
                    &XLSREAD_ERROR_SHEET,
                    format!("xlsread: unable to read sheet {}: {err:?}", index + 1),
                    err,
                )
            })?,
        None => workbook
            .worksheet_range_at(0)
            .ok_or_else(|| xlsread_error_with(&XLSREAD_ERROR_SHEET, "xlsread: no worksheets"))?
            .map_err(|err| {
                xlsread_error_with_source(
                    &XLSREAD_ERROR_SHEET,
                    format!("xlsread: unable to read first sheet: {err:?}"),
                    err,
                )
            })?,
    };
    let cells = selected_cells(&range, request.range, cooperation, path).await?;
    XlsReadResult::from_cells(cells, outputs, cooperation, path).await
}

async fn read_file_bytes(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
) -> BuiltinResult<Vec<u8>> {
    let file = await_io(runtime, BUILTIN_NAME, path.open_read())
        .await?
        .map_err(|err| {
            xlsread_error_with_source(
                &XLSREAD_ERROR_IO,
                format!("xlsread: unable to open '{}': {err}", path.path().display()),
                err,
            )
        })?;
    cooperation.completed(0).await?;
    path.check().map_err(|source| {
        xlsread_error_with_source(
            &XLSREAD_ERROR_IO,
            "xlsread: captured filesystem authority was revoked",
            source,
        )
    })?;
    let mut limited = file.take(MAX_XLSREAD_WORKBOOK_BYTES + 1);
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        let count = limited.read(&mut chunk).map_err(|err| {
            xlsread_error_with_source(
                &XLSREAD_ERROR_IO,
                format!("xlsread: unable to read '{}': {err}", path.path().display()),
                err,
            )
        })?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
        cooperation.completed(count).await?;
        path.check().map_err(|source| {
            xlsread_error_with_source(
                &XLSREAD_ERROR_IO,
                "xlsread: captured filesystem authority was revoked",
                source,
            )
        })?;
    }
    if bytes.len() as u64 > MAX_XLSREAD_WORKBOOK_BYTES {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_IO,
            format!(
                "xlsread: workbook exceeds maximum supported size of {MAX_XLSREAD_WORKBOOK_BYTES} bytes"
            ),
        ));
    }
    Ok(bytes)
}

pub(super) fn preflight_zip_workbook(bytes: &[u8]) -> BuiltinResult<()> {
    if !looks_like_zip(bytes) {
        return Ok(());
    }
    validate_zip_entry_count_from_raw(bytes)?;
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|err| {
        xlsread_error_with_source(
            &XLSREAD_ERROR_IO,
            format!("xlsread: invalid ZIP workbook: {err}"),
            err,
        )
    })?;
    validate_zip_entry_count(archive.len())?;
    let mut total = 0u64;
    for idx in 0..archive.len() {
        let entry = archive.by_index(idx).map_err(|err| {
            xlsread_error_with_source(
                &XLSREAD_ERROR_IO,
                format!("xlsread: unable to inspect ZIP workbook entry {idx}: {err}"),
                err,
            )
        })?;
        validate_zip_uncompressed_entry(entry.size(), &mut total)?;
    }
    Ok(())
}

fn looks_like_zip(bytes: &[u8]) -> bool {
    bytes.starts_with(b"PK\x03\x04")
        || bytes.starts_with(b"PK\x05\x06")
        || bytes.starts_with(b"PK\x07\x08")
}

pub(super) fn validate_zip_entry_count(entry_count: usize) -> BuiltinResult<()> {
    if entry_count > MAX_XLSREAD_ZIP_ENTRIES {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_IO,
            format!("xlsread: ZIP workbook contains more than {MAX_XLSREAD_ZIP_ENTRIES} entries"),
        ));
    }
    Ok(())
}

fn validate_zip_entry_count_from_raw(bytes: &[u8]) -> BuiltinResult<()> {
    let eocd_offset = find_zip_eocd(bytes).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_IO,
            "xlsread: invalid ZIP workbook: missing end of central directory",
        )
    })?;
    let total_entries = read_le_u16(bytes, eocd_offset + 10)? as usize;
    if total_entries == u16::MAX as usize {
        validate_zip64_entry_count_from_raw(bytes, eocd_offset)
    } else {
        validate_zip_entry_count(total_entries)
    }
}

fn validate_zip64_entry_count_from_raw(bytes: &[u8], eocd_offset: usize) -> BuiltinResult<()> {
    let locator_offset = eocd_offset.checked_sub(20).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_IO,
            "xlsread: invalid ZIP64 workbook: missing locator",
        )
    })?;
    if checked_zip_slice(bytes, locator_offset, 4)? != b"PK\x06\x07" {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_IO,
            "xlsread: invalid ZIP64 workbook: missing locator",
        ));
    }
    let zip64_locator_record_offset = locator_offset.checked_add(8).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_IO,
            "xlsread: invalid ZIP workbook: truncated directory",
        )
    })?;
    let zip64_eocd_offset = read_le_u64(bytes, zip64_locator_record_offset)?;
    let zip64_eocd_offset = usize::try_from(zip64_eocd_offset).map_err(|_| {
        xlsread_error_with(
            &XLSREAD_ERROR_IO,
            "xlsread: ZIP64 workbook directory offset exceeds supported bounds",
        )
    })?;
    if checked_zip_slice(bytes, zip64_eocd_offset, 4)? != b"PK\x06\x06" {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_IO,
            "xlsread: invalid ZIP64 workbook: missing end of central directory",
        ));
    }
    let total_entries_offset = zip64_eocd_offset.checked_add(32).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_IO,
            "xlsread: invalid ZIP workbook: truncated directory",
        )
    })?;
    let total_entries = read_le_u64(bytes, total_entries_offset)?;
    let total_entries = usize::try_from(total_entries).map_err(|_| {
        xlsread_error_with(
            &XLSREAD_ERROR_IO,
            "xlsread: ZIP workbook entry count exceeds supported bounds",
        )
    })?;
    validate_zip_entry_count(total_entries)
}

pub(super) fn find_zip_eocd(bytes: &[u8]) -> Option<usize> {
    const EOCD_LEN: usize = 22;
    const MAX_COMMENT_LEN: usize = u16::MAX as usize;
    if bytes.len() < EOCD_LEN {
        return None;
    }
    let min_offset = bytes.len().saturating_sub(EOCD_LEN + MAX_COMMENT_LEN);
    let max_offset = bytes.len() - EOCD_LEN;
    (min_offset..=max_offset).rev().find(|&offset| {
        if bytes.get(offset..offset + 4) != Some(b"PK\x05\x06") {
            return false;
        }
        let comment_len = u16::from_le_bytes([bytes[offset + 20], bytes[offset + 21]]) as usize;
        offset + EOCD_LEN + comment_len == bytes.len()
    })
}

pub(super) fn checked_zip_slice(bytes: &[u8], offset: usize, len: usize) -> BuiltinResult<&[u8]> {
    let end = offset.checked_add(len).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_IO,
            "xlsread: invalid ZIP workbook: truncated directory",
        )
    })?;
    bytes.get(offset..end).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_IO,
            "xlsread: invalid ZIP workbook: truncated directory",
        )
    })
}

fn read_le_u16(bytes: &[u8], offset: usize) -> BuiltinResult<u16> {
    let slice = checked_zip_slice(bytes, offset, 2)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_le_u64(bytes: &[u8], offset: usize) -> BuiltinResult<u64> {
    let slice = checked_zip_slice(bytes, offset, 8)?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

#[cfg(test)]
pub(super) fn validate_zip_uncompressed_limits<I>(sizes: I) -> BuiltinResult<()>
where
    I: IntoIterator<Item = u64>,
{
    let mut total = 0u64;
    let mut entry_count = 0usize;
    for size in sizes {
        entry_count = entry_count.checked_add(1).ok_or_else(|| {
            xlsread_error_with(
                &XLSREAD_ERROR_IO,
                "xlsread: ZIP workbook entry count overflows supported bounds",
            )
        })?;
        validate_zip_entry_count(entry_count)?;
        validate_zip_uncompressed_entry(size, &mut total)?;
    }
    Ok(())
}

fn validate_zip_uncompressed_entry(size: u64, total: &mut u64) -> BuiltinResult<()> {
    if size > MAX_XLSREAD_ZIP_ENTRY_UNCOMPRESSED_BYTES {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_IO,
            format!(
                "xlsread: ZIP workbook entry exceeds maximum expanded size of {MAX_XLSREAD_ZIP_ENTRY_UNCOMPRESSED_BYTES} bytes"
            ),
        ));
    }
    *total = total.checked_add(size).ok_or_else(|| {
        xlsread_error_with(
            &XLSREAD_ERROR_IO,
            "xlsread: ZIP workbook expanded size overflows supported bounds",
        )
    })?;
    if *total > MAX_XLSREAD_ZIP_TOTAL_UNCOMPRESSED_BYTES {
        return Err(xlsread_error_with(
            &XLSREAD_ERROR_IO,
            format!(
                "xlsread: ZIP workbook expanded size exceeds maximum of {MAX_XLSREAD_ZIP_TOTAL_UNCOMPRESSED_BYTES} bytes"
            ),
        ));
    }
    Ok(())
}
