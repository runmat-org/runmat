//! Checked native-width numeric, character, and index payload decoding.

use super::super::super::format::{
    MI_DOUBLE, MI_INT16, MI_INT32, MI_INT64, MI_INT8, MI_SINGLE, MI_UINT16, MI_UINT32, MI_UINT64,
    MI_UINT8, MI_UTF16, MI_UTF32, MI_UTF8,
};
use super::super::load_error;
use super::{Endian, TaggedData};
use crate::BuiltinResult;

pub(super) fn decode_numeric_values(elem: &TaggedData, endian: Endian) -> BuiltinResult<Vec<f64>> {
    let data = &elem.data;
    match elem.data_type {
        MI_INT8 => Ok(data.iter().map(|v| (*v as i8) as f64).collect()),
        MI_UINT8 => Ok(data.iter().map(|v| *v as f64).collect()),
        MI_INT16 => {
            ensure_data_width(data, 2, "numeric int16 data")?;
            Ok(data
                .chunks_exact(2)
                .map(|chunk| endian.read_i16(chunk) as f64)
                .collect())
        }
        MI_UINT16 => {
            ensure_data_width(data, 2, "numeric uint16 data")?;
            Ok(data
                .chunks_exact(2)
                .map(|chunk| endian.read_u16(chunk) as f64)
                .collect())
        }
        MI_INT32 => {
            ensure_data_width(data, 4, "numeric int32 data")?;
            Ok(data
                .chunks_exact(4)
                .map(|chunk| endian.read_i32(chunk) as f64)
                .collect())
        }
        MI_UINT32 => {
            ensure_data_width(data, 4, "numeric uint32 data")?;
            Ok(data
                .chunks_exact(4)
                .map(|chunk| endian.read_u32(chunk) as f64)
                .collect())
        }
        MI_SINGLE => {
            ensure_data_width(data, 4, "numeric single data")?;
            Ok(data
                .chunks_exact(4)
                .map(|chunk| endian.read_f32(chunk) as f64)
                .collect())
        }
        MI_DOUBLE => {
            ensure_data_width(data, 8, "numeric double data")?;
            Ok(data
                .chunks_exact(8)
                .map(|chunk| endian.read_f64(chunk))
                .collect())
        }
        MI_INT64 => {
            ensure_data_width(data, 8, "numeric int64 data")?;
            Ok(data
                .chunks_exact(8)
                .map(|chunk| endian.read_i64(chunk) as f64)
                .collect())
        }
        MI_UINT64 => {
            ensure_data_width(data, 8, "numeric uint64 data")?;
            Ok(data
                .chunks_exact(8)
                .map(|chunk| endian.read_u64(chunk) as f64)
                .collect())
        }
        _ => Err(load_error(format!(
            "load: unsupported numeric data type {}",
            elem.data_type
        ))),
    }
}

pub(super) fn decode_char_codes(elem: &TaggedData, endian: Endian) -> BuiltinResult<Vec<u16>> {
    match elem.data_type {
        MI_INT8 | MI_UINT8 | MI_UTF8 => {
            let text = String::from_utf8_lossy(&elem.data);
            Ok(text
                .chars()
                .map(|ch| {
                    if ch as u32 <= u16::MAX as u32 {
                        ch as u16
                    } else {
                        0xFFFD
                    }
                })
                .collect())
        }
        MI_UINT16 | MI_UTF16 => {
            ensure_data_width(&elem.data, 2, "character UTF-16 data")?;
            Ok(elem
                .data
                .chunks_exact(2)
                .map(|chunk| endian.read_u16(chunk))
                .collect())
        }
        MI_UTF32 => {
            ensure_data_width(&elem.data, 4, "character UTF-32 data")?;
            Ok(elem
                .data
                .chunks_exact(4)
                .map(|chunk| {
                    let code = endian.read_u32(chunk);
                    if code <= u16::MAX as u32 {
                        code as u16
                    } else {
                        0xFFFD
                    }
                })
                .collect())
        }
        _ => Err(load_error("load: unsupported character data encoding")),
    }
}

pub(super) fn decode_index_values(
    elem: &TaggedData,
    endian: Endian,
    label: &str,
) -> BuiltinResult<Vec<usize>> {
    match elem.data_type {
        MI_INT32 => {
            ensure_data_width(&elem.data, 4, label)?;
            let mut out = Vec::with_capacity(elem.data.len() / 4);
            for chunk in elem.data.chunks_exact(4) {
                let value = endian.read_i32(chunk);
                if value < 0 {
                    return Err(load_error(format!(
                        "load: {label} contain a negative index"
                    )));
                }
                out.push(value as usize);
            }
            Ok(out)
        }
        MI_UINT32 => {
            ensure_data_width(&elem.data, 4, label)?;
            Ok(elem
                .data
                .chunks_exact(4)
                .map(|chunk| endian.read_u32(chunk) as usize)
                .collect())
        }
        _ => Err(load_error(format!(
            "load: {label} must be MI_INT32/MI_UINT32"
        ))),
    }
}

pub(super) fn ensure_data_width(data: &[u8], width: usize, label: &str) -> BuiltinResult<()> {
    if !data.len().is_multiple_of(width) {
        return Err(load_error(format!("load: malformed {label}")));
    }
    Ok(())
}
