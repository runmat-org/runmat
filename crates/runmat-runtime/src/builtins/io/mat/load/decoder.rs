//! Level-5 MAT element decoder and value restoration.

use super::super::format::MatArray;
use super::super::format::{MAT_HEADER_LEN, MI_COMPRESSED, MI_MATRIX};
use super::{load_error, load_error_with_source, LOAD_ERROR_IO};
use crate::BuiltinResult;
use flate2::read::ZlibDecoder;
#[cfg(test)]
use runmat_value::Value;
use std::io::{Cursor, Read};

pub(super) use super::super::cooperation::MatCodecCooperation;

#[derive(Clone, Copy, Debug)]
enum Endian {
    Little,
    Big,
}

impl Endian {
    fn read_u16(self, bytes: &[u8]) -> u16 {
        match self {
            Endian::Little => u16::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => u16::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_i16(self, bytes: &[u8]) -> i16 {
        match self {
            Endian::Little => i16::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => i16::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_u32(self, bytes: &[u8]) -> u32 {
        match self {
            Endian::Little => u32::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => u32::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_i32(self, bytes: &[u8]) -> i32 {
        match self {
            Endian::Little => i32::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => i32::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_u64(self, bytes: &[u8]) -> u64 {
        match self {
            Endian::Little => u64::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => u64::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_i64(self, bytes: &[u8]) -> i64 {
        match self {
            Endian::Little => i64::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => i64::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_f32(self, bytes: &[u8]) -> f32 {
        match self {
            Endian::Little => f32::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => f32::from_be_bytes(bytes.try_into().unwrap()),
        }
    }

    fn read_f64(self, bytes: &[u8]) -> f64 {
        match self {
            Endian::Little => f64::from_le_bytes(bytes.try_into().unwrap()),
            Endian::Big => f64::from_be_bytes(bytes.try_into().unwrap()),
        }
    }
}

#[cfg(test)]
pub(super) fn read_mat_reader<R: Read>(reader: &mut R) -> BuiltinResult<Vec<(String, Value)>> {
    read_mat_arrays(reader)?
        .into_iter()
        .map(|(name, array)| Ok((name, mat_array_to_value(array)?)))
        .collect()
}

pub(super) fn read_mat_arrays<R: Read>(reader: &mut R) -> BuiltinResult<Vec<(String, MatArray)>> {
    let endian = read_mat_header(reader)?;
    read_variables_from_elements(reader, endian)
}

fn read_mat_header<R: Read>(reader: &mut R) -> BuiltinResult<Endian> {
    let mut header = [0u8; MAT_HEADER_LEN];
    reader.read_exact(&mut header).map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to read MAT-file header: {err}"),
            err,
        )
    })?;

    let description = String::from_utf8_lossy(&header[..116]);
    if description.contains("MATLAB 7.3") || header.starts_with(b"\x89HDF\r\n\x1a\n") {
        return Err(load_error(
            "load: MATLAB v7.3 MAT-files are HDF5-backed and are not supported yet",
        ));
    }

    let endian = match (header[126], header[127]) {
        (b'I', b'M') => Endian::Little,
        (b'M', b'I') => Endian::Big,
        _ => return Err(load_error("load: file is not a MATLAB Level-5 MAT-file")),
    };
    Ok(endian)
}

fn read_variables_from_elements<R: Read>(
    reader: &mut R,
    endian: Endian,
) -> BuiltinResult<Vec<(String, MatArray)>> {
    let mut variables = Vec::new();
    while let Some(tagged) = read_tagged(reader, true, endian)? {
        match tagged.data_type {
            MI_MATRIX => {
                let parsed = parse_matrix(&tagged.data, endian)?;
                variables.push((parsed.name, parsed.array));
            }
            MI_COMPRESSED => {
                let mut decoder = ZlibDecoder::new(Cursor::new(tagged.data));
                let mut inflated = Vec::new();
                decoder.read_to_end(&mut inflated).map_err(|err| {
                    load_error_with_source(
                        &LOAD_ERROR_IO,
                        format!("load: failed to decompress MAT element: {err}"),
                        err,
                    )
                })?;
                let mut cursor = Cursor::new(inflated);
                variables.extend(read_variables_from_elements(&mut cursor, endian)?);
            }
            _ => continue,
        }
    }
    Ok(variables)
}

/// Decode opened MAT elements while charging real input/decompression work to
/// the invocation. A compressed stream is parsed directly rather than first
/// expanding its entire payload into an unbounded second byte vector.
#[cfg(test)]
pub(super) async fn read_mat_reader_cooperative<R: Read>(
    reader: &mut R,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<Vec<(String, Value)>> {
    read_mat_arrays_cooperative(reader, cooperation)
        .await?
        .into_iter()
        .map(|(name, array)| Ok((name, mat_array_to_value(array)?)))
        .collect()
}

pub(in crate::builtins::io::mat) async fn read_mat_arrays_cooperative<R: Read>(
    reader: &mut R,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<Vec<(String, MatArray)>> {
    cooperation.completed(0).await?;
    let endian = read_mat_header(reader)?;
    cooperation.completed(MAT_HEADER_LEN).await?;
    let mut streams: Vec<Box<dyn Read + '_>> = vec![Box::new(reader)];
    let mut variables = Vec::new();
    while let Some(stream) = streams.last_mut() {
        let Some(tagged) =
            tagged::read_tagged_cooperative(stream.as_mut(), true, endian, cooperation).await?
        else {
            streams.pop();
            continue;
        };
        match tagged.data_type {
            MI_MATRIX => {
                let parsed = parse_matrix(&tagged.data, endian)?;
                variables.push((parsed.name, parsed.array));
                cooperation.completed(tagged.data.len()).await?;
            }
            MI_COMPRESSED => {
                if streams.len() >= 64 {
                    return Err(load_error(
                        "load: compressed MAT element nesting is too deep",
                    ));
                }
                streams.push(Box::new(ZlibDecoder::new(Cursor::new(tagged.data))));
                cooperation.completed(1).await?;
            }
            _ => cooperation.completed(tagged.data.len()).await?,
        }
    }
    Ok(variables)
}

mod arrays;
use arrays::parse_matrix;
mod numeric;
use numeric::{decode_char_codes, decode_index_values, decode_numeric_values, ensure_data_width};
mod values;
pub(super) use values::mat_array_to_value;
mod tagged;
use tagged::{bytes_to_string, read_tagged, TaggedData};

#[cfg(test)]
#[path = "decoder/tests.rs"]
mod tests;
