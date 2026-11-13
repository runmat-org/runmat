//! Checked MAT element tags and padded payload reads.

use super::super::super::format::MI_COMPRESSED;
use super::super::{load_error_with_source, LOAD_ERROR_IO};
use super::Endian;
use super::MatCodecCooperation;
use crate::BuiltinResult;
use std::io::Read;

const READ_CHUNK_BYTES: usize = 64 * 1024;

pub(super) fn bytes_to_string(bytes: &[u8]) -> String {
    let trimmed = bytes
        .iter()
        .copied()
        .take_while(|b| *b != 0)
        .collect::<Vec<u8>>();
    String::from_utf8(trimmed).unwrap_or_default()
}

pub(super) struct TaggedData {
    pub(super) data_type: u32,
    pub(super) data: Vec<u8>,
}

enum TagHeader {
    Inline(TaggedData),
    Body {
        data_type: u32,
        length: usize,
        padding: usize,
    },
}

fn read_header<R: Read + ?Sized>(
    reader: &mut R,
    allow_eof: bool,
    endian: Endian,
) -> BuiltinResult<Option<TagHeader>> {
    let mut type_bytes = [0u8; 4];
    let first = loop {
        match reader.read(&mut type_bytes[..1]) {
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            result => break result,
        }
    };
    match first {
        Ok(0) if allow_eof => return Ok(None),
        Ok(0) => {
            let err = std::io::Error::from(std::io::ErrorKind::UnexpectedEof);
            return Err(load_error_with_source(
                &LOAD_ERROR_IO,
                format!("load: failed to read MAT element header: {err}"),
                err,
            ));
        }
        Ok(1) => {}
        Ok(_) => unreachable!("one-byte MAT tag read"),
        Err(err) => {
            return Err(load_error_with_source(
                &LOAD_ERROR_IO,
                format!("load: failed to read MAT element header: {err}"),
                err,
            ));
        }
    }
    reader.read_exact(&mut type_bytes[1..]).map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to read MAT element header: {err}"),
            err,
        )
    })?;

    if allow_eof && type_bytes == [0; 4] {
        return Ok(None);
    }

    let type_field = endian.read_u32(&type_bytes);
    let high = (type_field >> 16) & 0xFFFF;
    let low = type_field & 0xFFFF;
    let small = match endian {
        Endian::Little if high != 0 => Some((low, high as usize)),
        Endian::Big if high != 0 && low <= 4 => Some((high, low as usize)),
        _ => None,
    };

    if let Some((data_type, num_bytes)) = small {
        if num_bytes > 4 {
            return Err(super::super::load_error(
                "load: compact MAT element length exceeds its four-byte payload",
            ));
        }
        let mut inline = [0u8; 4];
        reader.read_exact(&mut inline).map_err(|err| {
            load_error_with_source(
                &LOAD_ERROR_IO,
                format!("load: failed to read compact MAT element: {err}"),
                err,
            )
        })?;
        let data = inline[..num_bytes].to_vec();
        Ok(Some(TagHeader::Inline(TaggedData { data_type, data })))
    } else {
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes).map_err(|err| {
            load_error_with_source(
                &LOAD_ERROR_IO,
                format!("load: failed to read MAT element length: {err}"),
                err,
            )
        })?;
        let length = endian.read_u32(&len_bytes) as usize;
        let padding = if type_field == MI_COMPRESSED {
            0
        } else {
            (8 - (length % 8)) % 8
        };
        Ok(Some(TagHeader::Body {
            data_type: type_field,
            length,
            padding,
        }))
    }
}

fn append_checked(data: &mut Vec<u8>, chunk: &[u8]) -> BuiltinResult<()> {
    data.try_reserve(chunk.len()).map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            "load: MAT element length cannot be allocated",
            err,
        )
    })?;
    data.extend_from_slice(chunk);
    Ok(())
}

fn read_body<R: Read + ?Sized>(reader: &mut R, length: usize) -> BuiltinResult<Vec<u8>> {
    let mut data = Vec::new();
    let mut chunk = vec![0u8; length.min(READ_CHUNK_BYTES)];
    while data.len() < length {
        let next = (length - data.len()).min(chunk.len());
        reader.read_exact(&mut chunk[..next]).map_err(|err| {
            load_error_with_source(
                &LOAD_ERROR_IO,
                format!("load: failed to read MAT element body: {err}"),
                err,
            )
        })?;
        append_checked(&mut data, &chunk[..next])?;
    }
    Ok(data)
}

fn read_padding<R: Read + ?Sized>(reader: &mut R, padding: usize) -> BuiltinResult<()> {
    if padding != 0 {
        let mut pad = [0u8; 7];
        reader.read_exact(&mut pad[..padding]).map_err(|err| {
            load_error_with_source(
                &LOAD_ERROR_IO,
                format!("load: failed to read MAT padding: {err}"),
                err,
            )
        })?;
    }
    Ok(())
}

pub(super) fn read_tagged<R: Read + ?Sized>(
    reader: &mut R,
    allow_eof: bool,
    endian: Endian,
) -> BuiltinResult<Option<TaggedData>> {
    let Some(header) = read_header(reader, allow_eof, endian)? else {
        return Ok(None);
    };
    match header {
        TagHeader::Inline(data) => Ok(Some(data)),
        TagHeader::Body {
            data_type,
            length,
            padding,
        } => {
            let data = read_body(reader, length)?;
            read_padding(reader, padding)?;
            Ok(Some(TaggedData { data_type, data }))
        }
    }
}

pub(super) async fn read_tagged_cooperative<R: Read + ?Sized>(
    reader: &mut R,
    allow_eof: bool,
    endian: Endian,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<Option<TaggedData>> {
    let Some(header) = read_header(reader, allow_eof, endian)? else {
        return Ok(None);
    };
    match header {
        TagHeader::Inline(data) => {
            cooperation.completed(8).await?;
            Ok(Some(data))
        }
        TagHeader::Body {
            data_type,
            length,
            padding,
        } => {
            let mut data = Vec::new();
            // Keep bulk I/O scratch storage off the executor stack. This
            // future is polled beneath the VM and runtime dispatch chain, so a
            // fixed 64-KiB local array consumes stack even for tiny elements.
            let mut chunk = vec![0u8; length.min(READ_CHUNK_BYTES)];
            while data.len() < length {
                let next = (length - data.len()).min(chunk.len());
                reader.read_exact(&mut chunk[..next]).map_err(|err| {
                    load_error_with_source(
                        &LOAD_ERROR_IO,
                        format!("load: failed to read MAT element body: {err}"),
                        err,
                    )
                })?;
                append_checked(&mut data, &chunk[..next])?;
                cooperation.completed(next).await?;
            }
            read_padding(reader, padding)?;
            cooperation.completed(8 + padding).await?;
            Ok(Some(TaggedData { data_type, data }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn clean_end_of_elements_is_distinct_from_a_truncated_tag() {
        assert!(read_tagged(&mut Cursor::new([]), true, Endian::Little)
            .expect("clean end")
            .is_none());
        for length in 1..4 {
            let mut reader = Cursor::new(vec![1u8; length]);
            let error = read_tagged(&mut reader, true, Endian::Little)
                .err()
                .expect("partial tag is malformed");
            assert_eq!(error.identifier(), Some("RunMat:load:Io"));
        }
    }

    #[test]
    fn oversized_compact_length_is_rejected_instead_of_truncated() {
        let mut reader = Cursor::new([1u8, 0, 5, 0, 9, 8, 7, 6]);
        assert!(read_tagged(&mut reader, false, Endian::Little).is_err());
    }
}
