use super::scalars::*;
use super::*;

fn canonical_dims(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => shape.to_vec(),
    }
}

#[cfg(test)]
pub(super) fn convert_value(value: Value) -> LocalBoxFuture<'static, BuiltinResult<MatArray>> {
    convert_value_with_cooperation(value, MatEncodingOwner::Save, None)
}

async fn gather_with_cooperation(
    value: &Value,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<Value> {
    match cooperation.as_deref_mut() {
        Some(cooperation) => cooperation.await_io(gather_if_needed_async(value)).await?,
        None => gather_if_needed_async(value).await,
    }
}

pub(in crate::builtins::io::mat) fn convert_value_with_cooperation<'a, 'b>(
    value: Value,
    owner: MatEncodingOwner,
    mut cooperation: Option<&'a mut MatCodecCooperation<'b>>,
) -> LocalBoxFuture<'a, BuiltinResult<MatArray>> {
    // The codec future owns the input Value across suspension. Lease managed
    // descendants before returning it, including if the caller queues or
    // drops the future without ever polling it.
    let input_root = crate::execution::RootedValueLease::for_traced(&value)
        .map_err(|error| error.into_runtime_error());
    Box::pin(async move {
        let _input_root = input_root?;
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
                let dims = canonical_dims(t.shape());
                let storage = t
                    .into_numeric_storage()
                    .map_err(|error| owner.error(MatEncodingFailure::Io, &error))?;
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
                let dims = canonical_dims(t.shape());
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
                        let class = integer_storage_mat_class(storage.real());
                        let (real, imag) = storage.into_unbounded_components();
                        (
                            class,
                            NumericStorage::from(real),
                            NumericStorage::from(imag),
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
                if sparse.is_complex() {
                    return Err(owner.error(
                        MatEncodingFailure::Unsupported,
                        "complex sparse MAT arrays are not supported",
                    ));
                }
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
                    ca.shape.as_slice().to_vec()
                };
                Ok(MatArray {
                    class: MatClass::Char,
                    dims,
                    data: MatData::Char { data },
                })
            }
            Value::String(runmat_value::StringElement::Text(s)) => {
                let data = text_to_utf16_cooperative(&s, owner, &mut cooperation).await?;
                Ok(MatArray {
                    class: MatClass::Char,
                    dims: vec![1, data.len()],
                    data: MatData::Char { data },
                })
            }
            Value::StringArray(sa) => {
                if sa.data().len() == 1 {
                    return convert_value_with_cooperation(
                        Value::String(sa.data()[0].clone()),
                        owner,
                        cooperation.as_deref_mut(),
                    )
                    .await;
                }
                let mut elements = Vec::with_capacity(sa.data().len());
                for text in sa.data() {
                    let text = text.as_str().ok_or_else(|| owner.error(MatEncodingFailure::Unsupported, "typed missing strings are not representable in this MAT character encoding"))?;
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
                let mut elements = Vec::with_capacity(cell.data().len());
                for element in cell.iter_column_major() {
                    completed(&mut cooperation, 1).await?;
                    let gathered = gather_with_cooperation(element, &mut cooperation).await?;
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
                let field_names: Vec<String> =
                    struct_value.fields.keys().map(ToOwned::to_owned).collect();
                let mut field_values = Vec::with_capacity(field_names.len());
                for field in &field_names {
                    completed(&mut cooperation, field.len()).await?;
                    let val = struct_value.fields.get(field).ok_or_else(|| {
                        owner.error(
                            MatEncodingFailure::Selection,
                            format!("missing struct field '{field}'"),
                        )
                    })?;
                    let gathered = gather_with_cooperation(val, &mut cooperation).await?;
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
                let field_names = array
                    .field_names()
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>();
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
                        let gathered = gather_with_cooperation(value, &mut cooperation).await?;
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
            Value::Object(object) => {
                let class_name = object.class_name().display_name().to_owned();
                let saved = prepare_object_element(Value::Object(object), &mut cooperation).await?;
                let payload = saved
                    .with_owned_async(|value| {
                        convert_value_with_cooperation(value, owner, cooperation.as_deref_mut())
                    })
                    .await?;
                Ok(MatArray {
                    class: MatClass::Object,
                    dims: vec![1, 1],
                    data: MatData::Object {
                        class_name,
                        field_names: vec![
                            "__runmat_saved_kind".to_owned(),
                            "__runmat_saved_payload".to_owned(),
                        ],
                        field_values: vec![object_kind(false), payload],
                    },
                })
            }
            Value::HandleObject(handle) => {
                let class_name = handle.class_name.display_name().to_owned();
                let saved =
                    prepare_object_element(Value::HandleObject(handle), &mut cooperation).await?;
                let payload = saved
                    .with_owned_async(|value| {
                        convert_value_with_cooperation(value, owner, cooperation.as_deref_mut())
                    })
                    .await?;
                Ok(MatArray {
                    class: MatClass::Object,
                    dims: vec![1, 1],
                    data: MatData::Object {
                        class_name,
                        field_names: vec![
                            "__runmat_saved_kind".to_owned(),
                            "__runmat_saved_payload".to_owned(),
                        ],
                        field_values: vec![object_kind(true), payload],
                    },
                })
            }
            Value::ObjectArray(array) => {
                let class_name = array.class_name().display_name().to_owned();
                let dims = canonical_dims(array.shape());
                let data = array.data();
                let mut kinds = Vec::with_capacity(data.len());
                let mut payloads = Vec::with_capacity(data.len());
                for element in data.iter().cloned() {
                    completed(&mut cooperation, 1).await?;
                    let is_handle = matches!(element, Value::HandleObject(_));
                    kinds.push(object_kind(is_handle));
                    let saved = prepare_object_element(element, &mut cooperation).await?;
                    payloads.push(
                        saved
                            .with_owned_async(|value| {
                                convert_value_with_cooperation(
                                    value,
                                    owner,
                                    cooperation.as_deref_mut(),
                                )
                            })
                            .await?,
                    );
                }
                kinds.extend(payloads);
                Ok(MatArray {
                    class: MatClass::Object,
                    dims,
                    data: MatData::Object {
                        class_name,
                        field_names: vec![
                            "__runmat_saved_kind".to_owned(),
                            "__runmat_saved_payload".to_owned(),
                        ],
                        field_values: kinds,
                    },
                })
            }
            Value::GpuTensor(handle) => {
                let gathered =
                    gather_with_cooperation(&Value::GpuTensor(handle), &mut cooperation).await?;
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

fn object_kind(handle: bool) -> MatArray {
    MatArray {
        class: MatClass::Logical,
        dims: vec![1, 1],
        data: MatData::Logical {
            data: vec![u8::from(handle)],
        },
    }
}

async fn prepare_object_element(
    element: Value,
    cooperation: &mut Option<&mut MatCodecCooperation<'_>>,
) -> BuiltinResult<crate::execution::RootedValue> {
    match cooperation.as_deref_mut() {
        Some(cooperation) => {
            let runtime = cooperation.runtime();
            cooperation
                .await_io(
                    crate::builtins::introspection::object_serialization::prepare_mat_object_element(
                        element, runtime,
                    ),
                )
                .await?
        }
        None => {
            crate::builtins::introspection::object_serialization::prepare_mat_object_element(
                element, None,
            )
            .await
        }
    }
}
