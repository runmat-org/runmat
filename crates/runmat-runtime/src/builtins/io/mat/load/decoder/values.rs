//! Decoded MAT arrays to RunMat values, preserving exact classes and layout.

use super::super::super::format::{MatArray, MatClass, MatData};
use super::super::load_error;
use crate::BuiltinResult;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
    LogicalArray, NumericStorage, SparseTensor, StringArray, StructArray, Tensor, Value,
};

pub(in super::super) fn mat_array_to_value(array: MatArray) -> BuiltinResult<Value> {
    match array.data {
        MatData::Numeric { real, imag } => {
            let len = real.len();
            if let Some(imag) = imag {
                if imag.len() != len || imag.numeric_dtype() != real.numeric_dtype() {
                    return Err(load_error(
                        "load: complex data has mismatched real/imaginary components",
                    ));
                }
                return match (real, imag) {
                    (NumericStorage::F64(real), NumericStorage::F64(imag)) => {
                        if len == 1 {
                            Ok(Value::Complex(real[0], imag[0]))
                        } else {
                            let pairs = real.into_iter().zip(imag).collect();
                            ComplexTensor::new(pairs, array.dims)
                                .map(Value::ComplexTensor)
                                .map_err(|error| load_error(format!("load: {error}")))
                        }
                    }
                    (NumericStorage::F32(real), NumericStorage::F32(imag)) => {
                        let pairs = real.into_iter().zip(imag).collect();
                        ComplexTensor::from_f32(pairs, array.dims)
                            .map(Value::ComplexTensor)
                            .map_err(|error| load_error(format!("load: {error}")))
                    }
                    (real, imag) => {
                        let real = real
                            .into_integer_storage()
                            .map_err(|_| load_error("load: unsupported complex numeric storage"))?;
                        let imag = imag
                            .into_integer_storage()
                            .map_err(|_| load_error("load: unsupported complex numeric storage"))?;
                        let storage = IntegerComplexStorage::new(real, imag)
                            .map_err(|error| load_error(format!("load: {error}")))?;
                        ComplexTensor::new_integer(storage, array.dims)
                            .map(Value::ComplexTensor)
                            .map_err(|error| load_error(format!("load: {error}")))
                    }
                };
            }
            match real {
                NumericStorage::F64(real) if len == 1 => Ok(Value::Num(real[0])),
                NumericStorage::F64(real) => Tensor::new(real, array.dims)
                    .map(Value::Tensor)
                    .map_err(|error| load_error(format!("load: {error}"))),
                NumericStorage::F32(real) => Tensor::from_f32(real, array.dims)
                    .map(Value::Tensor)
                    .map_err(|error| load_error(format!("load: {error}"))),
                storage => {
                    let storage = storage
                        .into_integer_storage()
                        .map_err(|_| load_error("load: unsupported real numeric storage"))?;
                    if len == 1 {
                        Ok(Value::Int(integer_storage_scalar(&storage, 0)))
                    } else {
                        Tensor::new_integer(storage, array.dims)
                            .map(Value::Tensor)
                            .map_err(|error| load_error(format!("load: {error}")))
                    }
                }
            }
        }
        MatData::Logical { data } => {
            let total = if array.dims.contains(&0) {
                0
            } else {
                array
                    .dims
                    .iter()
                    .try_fold(1usize, |acc, d| acc.checked_mul(*d))
                    .ok_or_else(|| {
                        load_error("load: logical array shape exceeds platform limits")
                    })?
            };
            if data.len() != total {
                return Err(load_error("load: logical data length mismatch"));
            }
            if total == 1 {
                Ok(Value::Bool(data.first().copied().unwrap_or(0) != 0))
            } else {
                let logical = LogicalArray::new(data, array.dims.clone())
                    .map_err(|e| load_error(format!("load: {e}")))?;
                Ok(Value::LogicalArray(logical))
            }
        }
        MatData::Char { data } => {
            let unit_count = if array.dims.contains(&0) {
                0
            } else {
                array
                    .dims
                    .iter()
                    .try_fold(1usize, |count, &extent| count.checked_mul(extent))
                    .ok_or_else(|| load_error("load: character shape exceeds platform limits"))?
            };
            if data.len() != unit_count {
                return Err(load_error("load: character data length mismatch"));
            }
            let code_units = data
                .into_iter()
                .map(runmat_value::Utf16CodeUnit::from_raw)
                .collect();
            let char_array = CharArray::from_column_major(code_units, array.dims)
                .map_err(|e| load_error(format!("load: {e}")))?;
            Ok(Value::CharArray(char_array))
        }
        MatData::Cell { elements } => {
            if let Some(strings) = cell_elements_to_strings(&elements) {
                let string_array = StringArray::new(strings, array.dims.clone())
                    .map_err(|e| load_error(format!("load: {e}")))?;
                return Ok(Value::StringArray(string_array));
            }
            let expected = array
                .dims
                .iter()
                .try_fold(1usize, |product, &dim| product.checked_mul(dim))
                .ok_or_else(|| load_error("load: cell array shape exceeds platform limits"))?;
            if elements.len() != expected {
                return Err(load_error("load: cell array element count mismatch"));
            }
            let mut converted = Vec::with_capacity(elements.len());
            for elem in elements {
                converted.push(mat_array_to_value(elem)?);
            }
            CellArray::from_column_major(converted, array.dims.clone())
                .map(Value::Cell)
                .map_err(|err| load_error(format!("load: {err}")))
        }
        MatData::Struct {
            field_names,
            field_values,
        } => {
            let element_count = array
                .dims
                .iter()
                .try_fold(1usize, |product, extent| product.checked_mul(*extent))
                .ok_or_else(|| load_error("load: struct array shape exceeds platform limits"))?;
            let expected = field_names
                .len()
                .checked_mul(element_count)
                .ok_or_else(|| load_error("load: struct field storage exceeds platform limits"))?;
            if field_values.len() != expected {
                return Err(load_error("load: struct field metadata is inconsistent"));
            }
            let values = field_values
                .into_iter()
                .map(mat_array_to_value)
                .collect::<BuiltinResult<Vec<_>>>()?;
            StructArray::normalize_field_major(field_names, values, array.dims.clone())
                .map_err(load_error)
        }
        MatData::Object { .. } => Err(load_error(
            "load: object MAT node requires selected-runtime restoration before Value construction",
        )),
        MatData::Sparse {
            rows,
            cols,
            col_ptrs,
            row_indices,
            values,
            logical,
        } => {
            let sparse = if logical {
                let mut filtered_col_ptrs = Vec::with_capacity(cols.saturating_add(1));
                let mut filtered_row_indices = Vec::new();
                filtered_col_ptrs.push(0);
                for col in 0..cols {
                    for index in col_ptrs[col]..col_ptrs[col + 1] {
                        let value = values.value_at(index).ok_or_else(|| {
                            load_error("load: logical sparse value count does not match structure")
                        })?;
                        if !value.is_zero() {
                            filtered_row_indices.push(row_indices[index]);
                        }
                    }
                    filtered_col_ptrs.push(filtered_row_indices.len());
                }
                SparseTensor::new_logical(rows, cols, filtered_col_ptrs, filtered_row_indices)
            } else {
                match values {
                    NumericStorage::F64(values) => {
                        SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
                    }
                    NumericStorage::F32(values) => {
                        SparseTensor::new_f32(rows, cols, col_ptrs, row_indices, values)
                    }
                    storage => SparseTensor::new_integer(
                        rows,
                        cols,
                        col_ptrs,
                        row_indices,
                        storage
                            .into_integer_storage()
                            .expect("non-floating sparse MAT storage is integer"),
                    ),
                }
            }
            .map_err(|e| load_error(format!("load: {e}")))?;
            Ok(Value::SparseTensor(sparse))
        }
    }
}

fn integer_storage_scalar(storage: &IntegerStorage, index: usize) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[index]),
        IntegerStorage::I16(values) => IntValue::I16(values[index]),
        IntegerStorage::I32(values) => IntValue::I32(values[index]),
        IntegerStorage::I64(values) => IntValue::I64(values[index]),
        IntegerStorage::U8(values) => IntValue::U8(values[index]),
        IntegerStorage::U16(values) => IntValue::U16(values[index]),
        IntegerStorage::U32(values) => IntValue::U32(values[index]),
        IntegerStorage::U64(values) => IntValue::U64(values[index]),
    }
}

fn cell_elements_to_strings(elements: &[MatArray]) -> Option<Vec<String>> {
    let mut strings = Vec::with_capacity(elements.len());
    for element in elements {
        if element.class != MatClass::Char {
            return None;
        }
        let rows = element.dims.first().copied().unwrap_or(1);
        if rows > 1 {
            return None;
        }
        match &element.data {
            MatData::Char { data } => strings.push(utf16_codes_to_string(data)),
            _ => return None,
        }
    }
    Some(strings)
}

fn utf16_codes_to_string(data: &[u16]) -> String {
    let end = data
        .iter()
        .rposition(|&code| code != 0)
        .map_or(0, |index| index + 1);
    String::from_utf16_lossy(&data[..end])
}
