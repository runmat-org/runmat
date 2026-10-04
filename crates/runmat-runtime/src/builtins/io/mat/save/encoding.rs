//! Level-5 MAT value conversion and byte encoding.

use super::super::format::{MatArray, MatClass, MatData};
use super::{save_error_with, SAVE_ERROR_IO, SAVE_ERROR_SELECTION, SAVE_ERROR_UNSUPPORTED};
use crate::builtins::introspection::object_serialization::prepare_value_for_mat_save;
use crate::{gather_if_needed_async, BuiltinResult};
use futures::future::LocalBoxFuture;
use runmat_value::{
    CharArray, ComplexStorage, IntValue, IntegerStorage, NumericDType, NumericStorage, Value,
};

pub(super) struct MatVar {
    pub(super) name: String,
    pub(super) array: MatArray,
}

fn canonical_dims(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => shape.to_vec(),
    }
}

pub(super) fn convert_value(value: Value) -> LocalBoxFuture<'static, BuiltinResult<MatArray>> {
    Box::pin(async move {
        match value {
            Value::Num(n) => Ok(MatArray {
                class: MatClass::Double,
                dims: vec![1, 1],
                data: MatData::Numeric {
                    real: NumericStorage::F64(vec![n]),
                    imag: None,
                },
            }),
            Value::Int(i) => {
                let class = int_value_mat_class(&i);
                Ok(MatArray {
                    class,
                    dims: vec![1, 1],
                    data: MatData::Numeric {
                        real: NumericStorage::from(integer_storage_from_scalar(i)),
                        imag: None,
                    },
                })
            }
            Value::Bool(b) => Ok(MatArray {
                class: MatClass::Logical,
                dims: vec![1, 1],
                data: MatData::Logical {
                    data: vec![if b { 1 } else { 0 }],
                },
            }),
            Value::Tensor(t) => {
                let dims = canonical_dims(&t.shape);
                let storage = t
                    .into_numeric_storage()
                    .map_err(|error| save_error_with(&SAVE_ERROR_IO, format!("save: {error}")))?;
                let class = tensor_dtype_mat_class(storage.numeric_dtype());
                let data = MatData::Numeric {
                    real: storage,
                    imag: None,
                };
                Ok(MatArray { class, dims, data })
            }
            Value::Complex(re, im) => Ok(MatArray {
                class: MatClass::Double,
                dims: vec![1, 1],
                data: MatData::Numeric {
                    real: NumericStorage::F64(vec![re]),
                    imag: Some(NumericStorage::F64(vec![im])),
                },
            }),
            Value::ComplexTensor(t) => {
                let dims = canonical_dims(&t.shape);
                let (class, real, imag) = match t.into_complex_storage() {
                    ComplexStorage::F64(values) => {
                        let (real, imag): (Vec<_>, Vec<_>) = values.into_iter().unzip();
                        (
                            MatClass::Double,
                            NumericStorage::F64(real),
                            NumericStorage::F64(imag),
                        )
                    }
                    ComplexStorage::F32(values) => {
                        let (real, imag): (Vec<_>, Vec<_>) = values.into_iter().unzip();
                        (
                            MatClass::Single,
                            NumericStorage::F32(real),
                            NumericStorage::F32(imag),
                        )
                    }
                    ComplexStorage::Integer(storage) => {
                        let class = integer_storage_mat_class(&storage.real);
                        (
                            class,
                            NumericStorage::from(storage.real),
                            NumericStorage::from(storage.imag),
                        )
                    }
                };
                Ok(MatArray {
                    class,
                    dims,
                    data: MatData::Numeric {
                        real,
                        imag: Some(imag),
                    },
                })
            }
            Value::LogicalArray(la) => Ok(MatArray {
                class: MatClass::Logical,
                dims: canonical_dims(&la.shape),
                data: MatData::Logical {
                    data: la.data.into_vec(),
                },
            }),
            Value::SparseTensor(sparse) => {
                let logical = sparse.is_logical();
                let values = if logical {
                    NumericStorage::U8(vec![1; sparse.nnz()])
                } else if let Some(storage) = sparse.integer_storage() {
                    NumericStorage::from(storage.clone())
                } else if let Some(values) = sparse.as_f32_slice() {
                    NumericStorage::F32(values.to_vec())
                } else {
                    NumericStorage::F64(
                        sparse
                            .as_f64_slice()
                            .expect("double sparse storage must expose f64 values")
                            .to_vec(),
                    )
                };
                Ok(MatArray {
                    class: MatClass::Sparse,
                    dims: vec![sparse.rows, sparse.cols],
                    data: MatData::Sparse {
                        rows: sparse.rows,
                        cols: sparse.cols,
                        col_ptrs: sparse.col_ptrs.into_vec(),
                        row_indices: sparse.row_indices.into_vec(),
                        values,
                        logical,
                    },
                })
            }
            Value::CharArray(ca) => Ok(MatArray {
                class: MatClass::Char,
                dims: ca.shape.clone(),
                data: MatData::Char {
                    data: char_array_to_utf16(&ca),
                },
            }),
            Value::String(s) => Ok(MatArray {
                class: MatClass::Char,
                dims: vec![1, s.chars().count()],
                data: MatData::Char {
                    data: s.encode_utf16().collect(),
                },
            }),
            Value::StringArray(sa) => {
                if sa.data.len() == 1 {
                    return convert_value(Value::String(sa.data[0].clone())).await;
                }
                let mut elements = Vec::with_capacity(sa.data.len());
                for text in &sa.data {
                    elements.push(MatArray {
                        class: MatClass::Char,
                        dims: vec![1, text.chars().count()],
                        data: MatData::Char {
                            data: text.encode_utf16().collect(),
                        },
                    });
                }
                Ok(MatArray {
                    class: MatClass::Cell,
                    dims: canonical_dims(&sa.shape),
                    data: MatData::Cell { elements },
                })
            }
            Value::Cell(cell) => {
                let mut elements = Vec::with_capacity(cell.data.len());
                for element in cell.to_column_major() {
                    let gathered = gather_if_needed_async(&element).await?;
                    elements.push(convert_value(gathered).await?);
                }
                Ok(MatArray {
                    class: MatClass::Cell,
                    dims: canonical_dims(&cell.shape),
                    data: MatData::Cell { elements },
                })
            }
            Value::Struct(struct_value) => {
                let field_names: Vec<String> = struct_value.fields.keys().cloned().collect();
                let mut field_values = Vec::with_capacity(field_names.len());
                for field in &field_names {
                    let val = struct_value.fields.get(field).ok_or_else(|| {
                        save_error_with(
                            &SAVE_ERROR_SELECTION,
                            format!("save: missing struct field '{field}'"),
                        )
                    })?;
                    let gathered = gather_if_needed_async(val).await?;
                    field_values.push(convert_value(gathered).await?);
                }
                Ok(MatArray {
                    class: MatClass::Struct,
                    dims: vec![1, 1],
                    data: MatData::Struct {
                        field_names,
                        field_values,
                    },
                })
            }
            Value::StructArray(array) => {
                let field_names = array.field_names().cloned().collect::<Vec<_>>();
                let capacity = array.len().checked_mul(field_names.len()).ok_or_else(|| {
                    save_error_with(
                        &SAVE_ERROR_SELECTION,
                        "save: structure array exceeds platform limits",
                    )
                })?;
                let mut field_values = Vec::with_capacity(capacity);
                for field in &field_names {
                    let column = array.field_values(field).ok_or_else(|| {
                        save_error_with(
                            &SAVE_ERROR_SELECTION,
                            format!("save: missing struct field '{field}'"),
                        )
                    })?;
                    for value in column {
                        field_values
                            .push(convert_value(gather_if_needed_async(value).await?).await?);
                    }
                }
                Ok(MatArray {
                    class: MatClass::Struct,
                    dims: canonical_dims(array.shape()),
                    data: MatData::Struct {
                        field_names,
                        field_values,
                    },
                })
            }
            Value::GpuTensor(handle) => {
                let gathered = gather_if_needed_async(&Value::GpuTensor(handle)).await?;
                convert_value(gathered).await
            }
            unsupported => Err(save_error_with(
                &SAVE_ERROR_UNSUPPORTED,
                format!("save: value of type '{:?}' is not supported", unsupported),
            )),
        }
    })
}

fn char_array_to_utf16(ca: &CharArray) -> Vec<u16> {
    ca.to_column_major()
        .into_iter()
        .map(|value| value as u16)
        .collect()
}

fn int_value_mat_class(value: &IntValue) -> MatClass {
    match value {
        IntValue::I8(_) => MatClass::Int8,
        IntValue::I16(_) => MatClass::Int16,
        IntValue::I32(_) => MatClass::Int32,
        IntValue::I64(_) => MatClass::Int64,
        IntValue::U8(_) => MatClass::UInt8,
        IntValue::U16(_) => MatClass::UInt16,
        IntValue::U32(_) => MatClass::UInt32,
        IntValue::U64(_) => MatClass::UInt64,
    }
}

fn integer_storage_from_scalar(value: IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![value]),
    }
}

fn integer_storage_mat_class(storage: &IntegerStorage) -> MatClass {
    match storage {
        IntegerStorage::I8(_) => MatClass::Int8,
        IntegerStorage::I16(_) => MatClass::Int16,
        IntegerStorage::I32(_) => MatClass::Int32,
        IntegerStorage::I64(_) => MatClass::Int64,
        IntegerStorage::U8(_) => MatClass::UInt8,
        IntegerStorage::U16(_) => MatClass::UInt16,
        IntegerStorage::U32(_) => MatClass::UInt32,
        IntegerStorage::U64(_) => MatClass::UInt64,
    }
}

fn tensor_dtype_mat_class(dtype: NumericDType) -> MatClass {
    match dtype {
        NumericDType::F64 => MatClass::Double,
        NumericDType::F32 => MatClass::Single,
        NumericDType::I8 => MatClass::Int8,
        NumericDType::I16 => MatClass::Int16,
        NumericDType::I32 => MatClass::Int32,
        NumericDType::I64 => MatClass::Int64,
        NumericDType::U8 => MatClass::UInt8,
        NumericDType::U16 => MatClass::UInt16,
        NumericDType::U32 => MatClass::UInt32,
        NumericDType::U64 => MatClass::UInt64,
    }
}

mod writer;
use writer::write_mat_bytes;
pub(super) use writer::write_mat_file;
pub async fn encode_workspace_to_mat_bytes(entries: &[(String, Value)]) -> BuiltinResult<Vec<u8>> {
    let mut mat_vars = Vec::with_capacity(entries.len());
    for (name, value) in entries {
        let value = prepare_value_for_mat_save(value.clone()).await?;
        mat_vars.push(MatVar {
            name: name.clone(),
            array: convert_value(value).await?,
        });
    }
    write_mat_bytes(&mat_vars)
}
