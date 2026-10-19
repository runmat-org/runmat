//! Level-5 matrix layouts and nested array payloads.

use super::super::super::format::{
    MatArray, MatClass, MatData, FLAG_COMPLEX, FLAG_LOGICAL, MI_DOUBLE, MI_INT16, MI_INT32,
    MI_INT64, MI_INT8, MI_MATRIX, MI_SINGLE, MI_UINT16, MI_UINT32, MI_UINT64, MI_UINT8, MI_UTF16,
};
use super::super::load_error;
use super::{
    bytes_to_string, decode_char_codes, decode_index_values, decode_numeric_values,
    ensure_data_width, read_tagged, Endian, TaggedData,
};
use crate::BuiltinResult;
use runmat_value::{IntegerStorage, NumericStorage};
use std::io::Cursor;

pub(super) struct ParsedMatrix {
    pub(super) name: String,
    pub(super) array: MatArray,
}

pub(super) fn parse_matrix(buffer: &[u8], endian: Endian) -> BuiltinResult<ParsedMatrix> {
    let mut cursor = Cursor::new(buffer);

    let flags = read_tagged(&mut cursor, false, endian)?
        .ok_or_else(|| load_error("load: matrix element missing array flags"))?;
    if flags.data_type != MI_UINT32 || flags.data.len() < 8 {
        return Err(load_error("load: invalid array flags block"));
    }
    let flags0 = endian.read_u32(&flags.data[0..4]);
    let class_code = flags0 & 0xFF;
    let mut class = MatClass::from_class_code(class_code)
        .ok_or_else(|| load_error("load: unsupported MATLAB class"))?;
    let is_logical = (flags0 & FLAG_LOGICAL) != 0;
    let has_imag = (flags0 & FLAG_COMPLEX) != 0;
    if class == MatClass::Object && (is_logical || has_imag) {
        return Err(load_error(
            "load: object MAT arrays cannot carry logical or complex flags",
        ));
    }
    if is_logical && class != MatClass::Sparse {
        class = MatClass::Logical;
    }

    let dims_elem = read_tagged(&mut cursor, false, endian)?
        .ok_or_else(|| load_error("load: matrix element missing dimensions"))?;
    if dims_elem.data_type != MI_INT32 {
        return Err(load_error("load: dimension block must use MI_INT32"));
    }
    if dims_elem.data.is_empty() || dims_elem.data.len() % 4 != 0 {
        return Err(load_error("load: malformed dimension block"));
    }
    let mut dims = Vec::with_capacity(dims_elem.data.len() / 4);
    for chunk in dims_elem.data.chunks_exact(4) {
        let value = endian.read_i32(chunk);
        if value < 0 {
            return Err(load_error("load: negative dimensions are not supported"));
        }
        dims.push(value as usize);
    }
    if dims.is_empty() {
        dims.push(1);
        dims.push(1);
    }
    shape_elements(&dims)?;

    let name_elem = read_tagged(&mut cursor, false, endian)?
        .ok_or_else(|| load_error("load: matrix element missing name"))?;
    let name = match name_elem.data_type {
        MI_INT8 | MI_UINT8 => bytes_to_string(&name_elem.data),
        MI_UINT16 | MI_UTF16 => {
            let mut bytes = Vec::with_capacity(name_elem.data.len());
            for chunk in name_elem.data.chunks_exact(2) {
                let code = endian.read_u16(chunk);
                if code == 0 {
                    break;
                }
                if let Some(ch) = char::from_u32(code as u32) {
                    bytes.push(ch);
                }
            }
            bytes.into_iter().collect()
        }
        _ => {
            return Err(load_error("load: unsupported array name encoding"));
        }
    };

    let array = match class {
        MatClass::Double => parse_numeric_array(&mut cursor, class, dims, has_imag, endian)?,
        MatClass::Single
        | MatClass::Int8
        | MatClass::UInt8
        | MatClass::Int16
        | MatClass::UInt16
        | MatClass::Int32
        | MatClass::UInt32
        | MatClass::Int64
        | MatClass::UInt64 => parse_numeric_array(&mut cursor, class, dims, has_imag, endian)?,
        MatClass::Logical => parse_logical_array(&mut cursor, dims, endian)?,
        MatClass::Char => parse_char_array(&mut cursor, dims, endian)?,
        MatClass::Cell => parse_cell_array(&mut cursor, dims, endian)?,
        MatClass::Struct => parse_struct(&mut cursor, dims, endian)?,
        MatClass::Object => parse_object(&mut cursor, dims, endian)?,
        MatClass::Sparse => parse_sparse_array(&mut cursor, dims, has_imag, is_logical, endian)?,
    };

    Ok(ParsedMatrix { name, array })
}

fn parse_numeric_array(
    cursor: &mut Cursor<&[u8]>,
    class: MatClass,
    dims: Vec<usize>,
    has_imag: bool,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    let real_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: numeric array missing real component"))?;
    let expected = shape_elements(&dims)?;
    ensure_numeric_count(&real_elem, expected)?;
    if class_is_integer(class) {
        let storage = decode_integer_storage(&real_elem, class, endian)?;
        let imag = if has_imag {
            let imag_elem = read_tagged(cursor, false, endian)?.ok_or_else(|| {
                load_error("load: complex integer array missing imaginary component")
            })?;
            ensure_numeric_count(&imag_elem, expected)?;
            Some(decode_integer_storage(&imag_elem, class, endian)?)
        } else {
            None
        };
        return Ok(MatArray {
            class,
            dims,
            data: MatData::Numeric {
                real: storage.into(),
                imag: imag.map(Into::into),
            },
        });
    }

    let real = decode_numeric_storage(&real_elem, class, endian)?;

    let imag = if has_imag {
        let imag_elem = read_tagged(cursor, false, endian)?
            .ok_or_else(|| load_error("load: numeric array missing imaginary component"))?;
        ensure_numeric_count(&imag_elem, expected)?;
        Some(decode_numeric_storage(&imag_elem, class, endian)?)
    } else {
        None
    };

    Ok(MatArray {
        class,
        dims,
        data: MatData::Numeric { real, imag },
    })
}

fn decode_numeric_storage(
    element: &TaggedData,
    class: MatClass,
    endian: Endian,
) -> BuiltinResult<NumericStorage> {
    let values = decode_numeric_values(element, endian)?;
    match class {
        MatClass::Double => Ok(NumericStorage::F64(values)),
        MatClass::Single => Ok(NumericStorage::F32(
            values.into_iter().map(|value| value as f32).collect(),
        )),
        _ => Err(load_error("load: unsupported floating numeric class")),
    }
}

fn class_is_integer(class: MatClass) -> bool {
    matches!(
        class,
        MatClass::Int8
            | MatClass::UInt8
            | MatClass::Int16
            | MatClass::UInt16
            | MatClass::Int32
            | MatClass::UInt32
            | MatClass::Int64
            | MatClass::UInt64
    )
}

fn decode_integer_storage(
    elem: &TaggedData,
    class: MatClass,
    endian: Endian,
) -> BuiltinResult<IntegerStorage> {
    let expected_type = match class {
        MatClass::Int8 => MI_INT8,
        MatClass::UInt8 => MI_UINT8,
        MatClass::Int16 => MI_INT16,
        MatClass::UInt16 => MI_UINT16,
        MatClass::Int32 => MI_INT32,
        MatClass::UInt32 => MI_UINT32,
        MatClass::Int64 => MI_INT64,
        MatClass::UInt64 => MI_UINT64,
        _ => return Err(load_error("load: expected an integer MAT class")),
    };
    if elem.data_type != expected_type {
        return Err(load_error(format!(
            "load: integer MAT class payload type {} does not match class {:?}",
            elem.data_type, class
        )));
    }

    let data = &elem.data;
    match class {
        MatClass::Int8 => Ok(IntegerStorage::I8(
            data.iter().map(|value| *value as i8).collect(),
        )),
        MatClass::UInt8 => Ok(IntegerStorage::U8(data.clone())),
        MatClass::Int16 => {
            ensure_data_width(data, 2, "integer int16 data")?;
            Ok(IntegerStorage::I16(
                data.chunks_exact(2)
                    .map(|chunk| endian.read_i16(chunk))
                    .collect(),
            ))
        }
        MatClass::UInt16 => {
            ensure_data_width(data, 2, "integer uint16 data")?;
            Ok(IntegerStorage::U16(
                data.chunks_exact(2)
                    .map(|chunk| endian.read_u16(chunk))
                    .collect(),
            ))
        }
        MatClass::Int32 => {
            ensure_data_width(data, 4, "integer int32 data")?;
            Ok(IntegerStorage::I32(
                data.chunks_exact(4)
                    .map(|chunk| endian.read_i32(chunk))
                    .collect(),
            ))
        }
        MatClass::UInt32 => {
            ensure_data_width(data, 4, "integer uint32 data")?;
            Ok(IntegerStorage::U32(
                data.chunks_exact(4)
                    .map(|chunk| endian.read_u32(chunk))
                    .collect(),
            ))
        }
        MatClass::Int64 => {
            ensure_data_width(data, 8, "integer int64 data")?;
            Ok(IntegerStorage::I64(
                data.chunks_exact(8)
                    .map(|chunk| endian.read_i64(chunk))
                    .collect(),
            ))
        }
        MatClass::UInt64 => {
            ensure_data_width(data, 8, "integer uint64 data")?;
            Ok(IntegerStorage::U64(
                data.chunks_exact(8)
                    .map(|chunk| endian.read_u64(chunk))
                    .collect(),
            ))
        }
        _ => Err(load_error("load: expected an integer MAT class")),
    }
}

fn parse_logical_array(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    let elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: logical array missing data block"))?;
    ensure_numeric_count(&elem, shape_elements(&dims)?)?;
    let data = decode_numeric_values(&elem, endian)?
        .into_iter()
        .map(|v| if v != 0.0 { 1 } else { 0 })
        .collect();
    Ok(MatArray {
        class: MatClass::Logical,
        dims,
        data: MatData::Logical { data },
    })
}

fn parse_char_array(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    let elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: character array missing data block"))?;
    let data = decode_char_codes(&elem, endian)?;
    Ok(MatArray {
        class: MatClass::Char,
        dims,
        data: MatData::Char { data },
    })
}

fn parse_cell_array(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    let total = shape_elements(&dims)?;
    ensure_remaining_element_headers(cursor, total)?;
    let mut elements = Vec::new();
    elements.try_reserve_exact(total).map_err(|error| {
        super::super::load_error_with_source(
            &runmat_builtins::LOAD_ERROR_IO,
            "load: cell element storage cannot be allocated",
            error,
        )
    })?;
    for _ in 0..total {
        let elem = read_tagged(cursor, false, endian)?
            .ok_or_else(|| load_error("load: cell element missing matrix payload"))?;
        if elem.data_type != MI_MATRIX {
            return Err(load_error("load: cell elements must be matrices"));
        }
        let parsed = parse_matrix(&elem.data, endian)?;
        elements.push(parsed.array);
    }
    Ok(MatArray {
        class: MatClass::Cell,
        dims,
        data: MatData::Cell { elements },
    })
}

fn parse_struct(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    let (field_names, field_values) = parse_fields(cursor, &dims, endian)?;
    Ok(MatArray {
        class: MatClass::Struct,
        dims,
        data: MatData::Struct {
            field_names,
            field_values,
        },
    })
}

fn parse_object(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    let class = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: object missing class name"))?;
    if !matches!(class.data_type, MI_INT8 | MI_UINT8) {
        return Err(load_error(
            "load: object class name must use MI_INT8/MI_UINT8",
        ));
    }
    let class_name = bytes_to_string(&class.data);
    if class_name.is_empty() {
        return Err(load_error("load: object class name is empty"));
    }
    let (field_names, field_values) = parse_fields(cursor, &dims, endian)?;
    Ok(MatArray {
        class: MatClass::Object,
        dims,
        data: MatData::Object {
            class_name,
            field_names,
            field_values,
        },
    })
}

fn parse_fields(
    cursor: &mut Cursor<&[u8]>,
    dims: &[usize],
    endian: Endian,
) -> BuiltinResult<(Vec<String>, Vec<MatArray>)> {
    let element_count = shape_elements(dims)?;
    let len_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: struct missing maximum field length specifier"))?;
    if len_elem.data_type != MI_INT32 || len_elem.data.len() != 4 {
        return Err(load_error("load: struct field length must be MI_INT32"));
    }
    let max_len = endian.read_i32(&len_elem.data[..4]);
    if max_len <= 0 {
        return Err(load_error("load: struct field length must be positive"));
    }

    let names_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: struct missing field name table"))?;
    if names_elem.data_type != MI_INT8 && names_elem.data_type != MI_UINT8 {
        return Err(load_error(
            "load: struct field names must be stored as MI_INT8/MI_UINT8",
        ));
    }
    if names_elem.data.len() % (max_len as usize) != 0 {
        return Err(load_error("load: malformed struct field name table"));
    }
    let field_count = names_elem.data.len() / (max_len as usize);
    let mut field_names = Vec::with_capacity(field_count);
    for i in 0..field_count {
        let start = i * (max_len as usize);
        let end = start + (max_len as usize);
        let slice = &names_elem.data[start..end];
        field_names.push(bytes_to_string(slice));
    }

    let value_count = field_count
        .checked_mul(element_count)
        .ok_or_else(|| load_error("load: struct array field count exceeds platform limits"))?;
    ensure_remaining_element_headers(cursor, value_count)?;
    let mut field_values = Vec::new();
    field_values
        .try_reserve_exact(value_count)
        .map_err(|error| {
            super::super::load_error_with_source(
                &runmat_builtins::LOAD_ERROR_IO,
                "load: struct field storage cannot be allocated",
                error,
            )
        })?;
    for _ in 0..value_count {
        let elem = read_tagged(cursor, false, endian)?
            .ok_or_else(|| load_error("load: struct field missing matrix payload"))?;
        if elem.data_type != MI_MATRIX {
            return Err(load_error("load: struct fields must be matrices"));
        }
        let parsed = parse_matrix(&elem.data, endian)?;
        field_values.push(parsed.array);
    }

    Ok((field_names, field_values))
}

fn shape_elements(dims: &[usize]) -> BuiltinResult<usize> {
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter()
        .try_fold(1usize, |product, extent| product.checked_mul(*extent))
        .ok_or_else(|| load_error("load: MAT array shape exceeds platform limits"))
}

fn ensure_numeric_count(element: &TaggedData, expected: usize) -> BuiltinResult<()> {
    let width = match element.data_type {
        MI_INT8 | MI_UINT8 => 1,
        MI_INT16 | MI_UINT16 => 2,
        MI_INT32 | MI_UINT32 | MI_SINGLE => 4,
        MI_INT64 | MI_UINT64 | MI_DOUBLE => 8,
        _ => return Err(load_error("load: unsupported numeric MAT payload type")),
    };
    if !element.data.len().is_multiple_of(width) || element.data.len() / width != expected {
        return Err(load_error(
            "load: numeric payload length does not match declared array shape",
        ));
    }
    Ok(())
}

fn ensure_remaining_element_headers(cursor: &Cursor<&[u8]>, count: usize) -> BuiltinResult<()> {
    let remaining = cursor
        .get_ref()
        .len()
        .saturating_sub(cursor.position() as usize);
    if count > remaining / 8 {
        return Err(load_error(
            "load: MAT array declares more elements than its payload contains",
        ));
    }
    Ok(())
}

fn parse_sparse_array(
    cursor: &mut Cursor<&[u8]>,
    dims: Vec<usize>,
    has_imag: bool,
    logical: bool,
    endian: Endian,
) -> BuiltinResult<MatArray> {
    if has_imag {
        return Err(load_error(
            "load: complex sparse MAT arrays are not supported yet",
        ));
    }
    let rows = dims.first().copied().unwrap_or(0);
    let cols = dims.get(1).copied().unwrap_or(0);

    let ir_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: sparse array missing row indices"))?;
    let row_indices = decode_index_values(&ir_elem, endian, "sparse row indices")?;

    let jc_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: sparse array missing column pointers"))?;
    let col_ptrs = decode_index_values(&jc_elem, endian, "sparse column pointers")?;

    let real_elem = read_tagged(cursor, false, endian)?
        .ok_or_else(|| load_error("load: sparse array missing values"))?;
    let values = match decode_sparse_integer_storage(&real_elem, endian)? {
        Some(storage) => NumericStorage::from(storage),
        None if real_elem.data_type == MI_SINGLE => NumericStorage::F32(
            decode_numeric_values(&real_elem, endian)?
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        ),
        None => NumericStorage::F64(decode_numeric_values(&real_elem, endian)?),
    };

    Ok(MatArray {
        class: MatClass::Sparse,
        dims,
        data: MatData::Sparse {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values,
            logical,
        },
    })
}

fn decode_sparse_integer_storage(
    elem: &TaggedData,
    endian: Endian,
) -> BuiltinResult<Option<IntegerStorage>> {
    let class = match elem.data_type {
        MI_INT8 => MatClass::Int8,
        MI_UINT8 => MatClass::UInt8,
        MI_INT16 => MatClass::Int16,
        MI_UINT16 => MatClass::UInt16,
        MI_INT32 => MatClass::Int32,
        MI_UINT32 => MatClass::UInt32,
        MI_INT64 => MatClass::Int64,
        MI_UINT64 => MatClass::UInt64,
        _ => return Ok(None),
    };
    decode_integer_storage(elem, class, endian).map(Some)
}

#[cfg(test)]
#[path = "arrays_geometry_tests.rs"]
mod geometry_tests;
