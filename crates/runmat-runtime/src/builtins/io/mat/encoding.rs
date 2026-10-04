//! Shared Level-5 MAT value conversion and byte encoding.

use super::cooperation::MatCodecCooperation;
use super::format::{MatArray, MatClass, MatData};

mod diagnostics;
use crate::builtins::introspection::object_serialization::prepare_value_for_mat_save;
use crate::{gather_if_needed_async, BuiltinResult};
use diagnostics::MatEncodingFailure;
pub(crate) use diagnostics::MatEncodingOwner;
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
    convert_value_with_cooperation(value, MatEncodingOwner::Save, None)
}

async fn completed(
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
    work: usize,
) -> BuiltinResult<()> {
    if let Some(cooperation) = cooperation.as_deref_mut() {
        cooperation.completed(work).await?;
    }
    Ok(())
}

pub(super) fn convert_value_with_cooperation<'a, 'b>(
    value: Value,
    owner: MatEncodingOwner,
    mut cooperation: Option<&'a mut MatCodecCooperation<'b>>,
) -> LocalBoxFuture<'a, BuiltinResult<MatArray>> {
    Box::pin(async move {
        completed(&mut cooperation, 0).await?;
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
                    .map_err(|error| owner.error(MatEncodingFailure::Io, error.to_string()))?;
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
            Value::CharArray(ca) => {
                let data = char_array_to_utf16_cooperative(&ca, owner, &mut cooperation).await?;
                let dims = if ca.shape.len() == 2 && ca.rows == 1 {
                    vec![1, data.len()]
                } else {
                    ca.shape.clone()
                };
                Ok(MatArray {
                    class: MatClass::Char,
                    dims,
                    data: MatData::Char { data },
                })
            }
            Value::String(s) => {
                let data = text_to_utf16_cooperative(&s, owner, &mut cooperation).await?;
                Ok(MatArray {
                    class: MatClass::Char,
                    dims: vec![1, data.len()],
                    data: MatData::Char { data },
                })
            }
            Value::StringArray(sa) => {
                if sa.data.len() == 1 {
                    return convert_value_with_cooperation(
                        Value::String(sa.data[0].clone()),
                        owner,
                        cooperation.as_deref_mut(),
                    )
                    .await;
                }
                let mut elements = Vec::with_capacity(sa.data.len());
                for text in &sa.data {
                    let data = text_to_utf16_cooperative(text, owner, &mut cooperation).await?;
                    elements.push(MatArray {
                        class: MatClass::Char,
                        dims: vec![1, data.len()],
                        data: MatData::Char { data },
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
                for element in cell.iter_column_major() {
                    completed(&mut cooperation, 1).await?;
                    let gathered = gather_if_needed_async(element).await?;
                    completed(&mut cooperation, 0).await?;
                    elements.push(
                        convert_value_with_cooperation(gathered, owner, cooperation.as_deref_mut())
                            .await?,
                    );
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
                    completed(&mut cooperation, field.len()).await?;
                    let val = struct_value.fields.get(field).ok_or_else(|| {
                        owner.error(
                            MatEncodingFailure::Selection,
                            format!("missing struct field '{field}'"),
                        )
                    })?;
                    let gathered = gather_if_needed_async(val).await?;
                    completed(&mut cooperation, 0).await?;
                    field_values.push(
                        convert_value_with_cooperation(gathered, owner, cooperation.as_deref_mut())
                            .await?,
                    );
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
                    owner.error(
                        MatEncodingFailure::Selection,
                        "structure array exceeds platform limits",
                    )
                })?;
                let mut field_values = Vec::with_capacity(capacity);
                for field in &field_names {
                    completed(&mut cooperation, field.len()).await?;
                    let column = array.field_values(field).ok_or_else(|| {
                        owner.error(
                            MatEncodingFailure::Selection,
                            format!("missing struct field '{field}'"),
                        )
                    })?;
                    for value in column {
                        completed(&mut cooperation, 1).await?;
                        let gathered = gather_if_needed_async(value).await?;
                        completed(&mut cooperation, 0).await?;
                        field_values.push(
                            convert_value_with_cooperation(
                                gathered,
                                owner,
                                cooperation.as_deref_mut(),
                            )
                            .await?,
                        );
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
                completed(&mut cooperation, 0).await?;
                convert_value_with_cooperation(gathered, owner, cooperation.as_deref_mut()).await
            }
            unsupported => Err(owner.error(
                MatEncodingFailure::Unsupported,
                format!("value of type '{unsupported:?}' is not supported"),
            )),
        }
    })
}

const TEXT_WORK_CHUNK: usize = 4096;

async fn text_to_utf16_cooperative(
    text: &str,
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<Vec<u16>> {
    let mut data = Vec::new();
    let mut work = 0usize;
    for unit in text.encode_utf16() {
        if data.capacity() == data.len() {
            data.try_reserve(TEXT_WORK_CHUNK).map_err(|error| {
                owner.error(
                    MatEncodingFailure::Io,
                    format!("UTF-16 data cannot be allocated: {error}"),
                )
            })?;
        }
        data.push(unit);
        work += 1;
        if work == TEXT_WORK_CHUNK {
            completed(cooperation, work).await?;
            work = 0;
        }
    }
    completed(cooperation, work).await?;
    Ok(data)
}

async fn char_array_to_utf16_cooperative(
    chars: &CharArray,
    owner: MatEncodingOwner,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<Vec<u16>> {
    if chars.shape.len() != 2 || chars.rows != 1 {
        if chars.data.iter().any(|ch| ch.len_utf16() != 1) {
            return Err(owner.error(
                MatEncodingFailure::Unsupported,
                "non-BMP multidimensional MAT character arrays cannot retain rectangular UTF-16 geometry",
            ));
        }
    }
    let mut data = Vec::new();
    let mut work = 0usize;
    let page_len = chars.rows.checked_mul(chars.cols).ok_or_else(|| {
        owner.error(
            MatEncodingFailure::Io,
            "MAT character shape exceeds platform limits",
        )
    })?;
    let pages = if page_len == 0 {
        0
    } else if chars.shape.len() <= 2 {
        1
    } else if chars.shape[2..].contains(&0) {
        0
    } else {
        chars.shape[2..]
            .iter()
            .try_fold(1usize, |product, &size| product.checked_mul(size))
            .ok_or_else(|| {
                owner.error(
                    MatEncodingFailure::Io,
                    "MAT character shape exceeds platform limits",
                )
            })?
    };
    if page_len.checked_mul(pages) != Some(chars.data.len()) {
        return Err(owner.error(
            MatEncodingFailure::Io,
            "MAT character payload does not match its shape",
        ));
    }
    for page in 0..pages {
        let page_offset = page * page_len;
        for col in 0..chars.cols {
            for row in 0..chars.rows {
                let mut units = [0; 2];
                for &unit in chars.data[page_offset + row * chars.cols + col]
                    .encode_utf16(&mut units)
                    .iter()
                {
                    if data.capacity() == data.len() {
                        data.try_reserve(TEXT_WORK_CHUNK).map_err(|error| {
                            owner.error(
                                MatEncodingFailure::Io,
                                format!("MAT characters cannot be allocated: {error}"),
                            )
                        })?;
                    }
                    data.push(unit);
                    work += 1;
                    if work == TEXT_WORK_CHUNK {
                        completed(cooperation, work).await?;
                        work = 0;
                    }
                }
            }
        }
    }
    completed(cooperation, work).await?;
    Ok(data)
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
use writer::write_mat_bytes_cooperative;
pub(super) use writer::write_mat_file;
pub async fn encode_workspace_to_mat_bytes(entries: &[(String, Value)]) -> BuiltinResult<Vec<u8>> {
    encode_workspace_to_mat_bytes_for_owner(entries, MatEncodingOwner::Save, None).await
}

pub(crate) async fn encode_workspace_to_mat_bytes_for_owner(
    entries: &[(String, Value)],
    owner: MatEncodingOwner,
    mut cooperation: Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<Vec<u8>> {
    let mut mat_vars = Vec::with_capacity(entries.len());
    for (name, value) in entries {
        completed(&mut cooperation, 0).await?;
        let value = match cooperation.as_deref_mut() {
            Some(cooperation) => {
                cooperation
                    .await_io(prepare_value_for_mat_save(value.clone()))
                    .await??
            }
            None => prepare_value_for_mat_save(value.clone()).await?,
        };
        completed(&mut cooperation, 0).await?;
        mat_vars.push(MatVar {
            name: name.clone(),
            array: convert_value_with_cooperation(value, owner, cooperation.as_deref_mut()).await?,
        });
    }
    write_mat_bytes_cooperative(&mat_vars, owner, cooperation).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use futures::executor::block_on;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_value::StringArray;
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn utf16_char_geometry_matches_emitted_code_units() {
        let array = block_on(convert_value(Value::String("A😀".into()))).expect("encode text");
        assert_eq!(array.dims, vec![1, 3]);
        let MatData::Char { data } = array.data else {
            panic!("text must become MAT char data");
        };
        assert_eq!(data, vec![0x0041, 0xd83d, 0xde00]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode_char_row_expands_mat_unit_geometry_without_truncation() {
        let source = CharArray::new_row("A😀");
        let array = block_on(convert_value(Value::CharArray(source)))
            .expect("Unicode char row has representable MAT geometry");
        assert_eq!(array.dims, vec![1, 3]);
        let MatData::Char { data } = array.data else {
            panic!("expected character payload");
        };
        assert_eq!(data, vec![0x0041, 0xd83d, 0xde00]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_bmp_multidimensional_char_array_rejects_instead_of_truncating() {
        let source = CharArray::new_with_shape(vec!['😀', 'x'], vec![2, 1])
            .expect("rectangular RunMat characters");
        let error = block_on(convert_value(Value::CharArray(source)))
            .expect_err("two-dimensional MAT units cannot represent scalar-width rows");
        assert_eq!(error.identifier(), Some("RunMat:save:Unsupported"));
        assert!(error.message().contains("UTF-16 geometry"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn matfile_encoding_uses_owning_error_identity() {
        let error = block_on(convert_value_with_cooperation(
            Value::OutputList(Vec::new()),
            MatEncodingOwner::MatFile,
            None,
        ))
        .expect_err("MAT-file assignment rejects unsupported payload");
        assert_eq!(error.identifier(), Some("RunMat:matfile:Unsupported"));
        assert!(error.message().starts_with("matfile:"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nested_string_cells_cancel_between_elements_without_result() {
        let cancellation = CancellationSource::new();
        let turns = Arc::new(AtomicUsize::new(0));
        let observed = turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            cancellation.token(),
            move || {
                let turn = observed.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    if turn == 0 {
                        Ok(())
                    } else {
                        Err(ExecutionServiceError::Cancelled)
                    }
                })
            },
        );
        let mut cooperation =
            MatCodecCooperation::with_budget(budget, "save", &runmat_builtins::SAVE_ERROR_IO);
        let strings = Value::StringArray(StringArray {
            data: vec!["first".into(), "second".into(), "third".into()],
            shape: vec![1, 3],
            rows: 1,
            cols: 3,
        });
        let error = block_on(convert_value_with_cooperation(
            strings,
            MatEncodingOwner::Save,
            Some(&mut cooperation),
        ))
        .expect_err("cancellation before later nested entries");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(turns.load(Ordering::SeqCst), 2);
        assert!(!cancellation.token().is_cancelled());
    }
}
