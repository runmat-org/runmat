use super::*;

fn reserved_values(len: usize) -> BuiltinResult<Vec<Complex<f64>>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    Ok(values)
}

fn reserved_columns(len: usize) -> BuiltinResult<Vec<Vec<Complex<f64>>>> {
    let mut columns = Vec::new();
    columns
        .try_reserve_exact(len)
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    Ok(columns)
}

/// Reject a definitely invalid signal before any resident control can be read.
/// Numeric contents are intentionally not constrained here.
pub(super) fn preflight_signal_carrier(value: &Value) -> BuiltinResult<bool> {
    let shape = match value {
        Value::GpuTensor(handle) => {
            gpu_matrix_shape(BUILTIN_NAME, "x", handle).map_err(|error| {
                pwelch_error_with_detail(&PWELCH_ERROR_INVALID_SIGNAL, error.message())
            })?;
            return Ok(runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
        }
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::ComplexTensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(array.shape.as_slice()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => None,
        _ => return Err(pwelch_error(&PWELCH_ERROR_INVALID_SIGNAL)),
    };
    if let Some(shape) = shape {
        if shape.len() > 2 || shape.contains(&0) {
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_SIGNAL));
        }
    }
    Ok(matches!(
        value,
        Value::Complex(_, _) | Value::ComplexTensor(_)
    ))
}

pub(super) async fn value_to_signal_columns(
    value: Value,
    prepared_integer: Option<crate::builtins::common::validation::PreparedIntegerPayload>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<SignalColumns> {
    cooperation.completed(0).await?;
    match value {
        Value::GpuTensor(handle) => {
            let value = match prepared_integer {
                Some(crate::builtins::common::validation::PreparedIntegerPayload::Real(
                    gathered,
                )) => Value::Tensor(
                    Tensor::new_integer(
                        crate::builtins::common::gpu_helpers::integer_storage_from_owned(
                            gathered.data,
                        ),
                        gathered.shape,
                    )
                    .map_err(|cause| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, cause))?,
                ),
                Some(crate::builtins::common::validation::PreparedIntegerPayload::Complex(_)) => {
                    return Err(pwelch_error(&PWELCH_ERROR_INVALID_SIGNAL));
                }
                None => {
                    let provider = super::super::resident_output::owner(&handle, || {
                        pwelch_error(&PWELCH_ERROR_INTERNAL)
                    })?;
                    crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
                        provider, &handle,
                    )
                    .await?
                }
            };
            cooperation.completed(0).await?;
            match value {
                Value::Tensor(tensor) => tensor_to_signal_columns(tensor, cooperation).await,
                Value::ComplexTensor(tensor) => {
                    complex_tensor_to_signal_columns(tensor, cooperation).await
                }
                Value::LogicalArray(logical) => {
                    let tensor = crate::builtins::common::tensor::logical_to_tensor(&logical)
                        .map_err(|err| {
                            pwelch_error_with_detail(&PWELCH_ERROR_INVALID_SIGNAL, err)
                        })?;
                    tensor_to_signal_columns(tensor, cooperation).await
                }
                Value::Num(value) => {
                    tensor_to_signal_columns(
                        Tensor::new(vec![value], handle.shape().to_vec()).map_err(|cause| {
                            pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, cause)
                        })?,
                        cooperation,
                    )
                    .await
                }
                other => Err(pwelch_error_with_detail(
                    &PWELCH_ERROR_INVALID_SIGNAL,
                    format!("unexpected downloaded signal {other:?}"),
                )),
            }
        }
        Value::Tensor(tensor) => tensor_to_signal_columns(tensor, cooperation).await,
        Value::ComplexTensor(tensor) => complex_tensor_to_signal_columns(tensor, cooperation).await,
        Value::LogicalArray(logical) => {
            let tensor = crate::builtins::common::tensor::logical_to_tensor(&logical)
                .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INVALID_SIGNAL, err))?;
            tensor_to_signal_columns(tensor, cooperation).await
        }
        Value::Num(n) => Ok(SignalColumns {
            columns: vec![vec![Complex::new(n, 0.0)]],
            rows: 1,
            cols: 1,
            is_complex: false,
            output_precision: SignalPrecision::Double,
        }),
        Value::Int(i) => Ok(SignalColumns {
            columns: vec![vec![Complex::new(i.to_f64(), 0.0)]],
            rows: 1,
            cols: 1,
            is_complex: false,
            output_precision: SignalPrecision::Double,
        }),
        Value::Bool(b) => Ok(SignalColumns {
            columns: vec![vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)]],
            rows: 1,
            cols: 1,
            is_complex: false,
            output_precision: SignalPrecision::Double,
        }),
        Value::Complex(re, im) => Ok(SignalColumns {
            columns: vec![vec![Complex::new(re, im)]],
            rows: 1,
            cols: 1,
            is_complex: true,
            output_precision: SignalPrecision::Double,
        }),
        other => Err(pwelch_error_with_detail(
            &PWELCH_ERROR_INVALID_SIGNAL,
            format!("unsupported input type {other:?}"),
        )),
    }
}

pub(super) async fn tensor_to_signal_columns(
    tensor: Tensor,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<SignalColumns> {
    if tensor.shape().len() > 2 {
        return Err(pwelch_error_with_detail(
            &PWELCH_ERROR_INVALID_SIGNAL,
            "x must be a vector or 2-D matrix",
        ));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    let output_precision = SignalPrecision::from_numeric_dtype(tensor.numeric_dtype());
    let mut work = CompletedWork::default();
    if rows == 1 || cols == 1 {
        let len = tensor_utils::tensor_element_len(&tensor);
        let mut values = reserved_values(len)?;
        for index in 0..len {
            values.push(Complex::new(
                tensor_utils::tensor_value_f64(&tensor, index),
                0.0,
            ));
            if let Some(units) = work.record(1) {
                cooperation.completed(units).await?;
            }
        }
        work.finish(cooperation).await?;
        return Ok(SignalColumns {
            columns: vec![values],
            rows: rows.max(cols),
            cols: 1,
            is_complex: false,
            output_precision,
        });
    }
    let mut columns = reserved_columns(cols)?;
    for col in 0..cols {
        let mut column = reserved_values(rows)?;
        for row in 0..rows {
            column.push(Complex::new(
                tensor_utils::tensor_value_f64(&tensor, row + col * rows),
                0.0,
            ));
            if let Some(units) = work.record(1) {
                cooperation.completed(units).await?;
            }
        }
        columns.push(column);
    }
    work.finish(cooperation).await?;
    Ok(SignalColumns {
        columns,
        rows,
        cols,
        is_complex: false,
        output_precision,
    })
}

pub(super) async fn complex_tensor_to_signal_columns(
    tensor: runmat_value::ComplexTensor,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<SignalColumns> {
    if tensor.shape().len() > 2 {
        return Err(pwelch_error_with_detail(
            &PWELCH_ERROR_INVALID_SIGNAL,
            "x must be a vector or 2-D matrix",
        ));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    let output_precision = SignalPrecision::from_numeric_dtype(tensor.numeric_dtype());
    let is_complex = true;
    let mut work = CompletedWork::default();
    if rows == 1 || cols == 1 {
        let len = rows.max(cols);
        let mut values = reserved_values(len)?;
        for index in 0..len {
            let (re, im) = tensor
                .numeric_value_at(index)
                .ok_or_else(|| pwelch_error(&PWELCH_ERROR_INTERNAL))?;
            values.push(Complex::new(scalar_f64(re), scalar_f64(im)));
            if let Some(units) = work.record(1) {
                cooperation.completed(units).await?;
            }
        }
        work.finish(cooperation).await?;
        return Ok(SignalColumns {
            columns: vec![values],
            rows: rows.max(cols),
            cols: 1,
            is_complex,
            output_precision,
        });
    }
    let mut columns = reserved_columns(cols)?;
    for col in 0..cols {
        let mut column = reserved_values(rows)?;
        for row in 0..rows {
            let (re, im) = tensor
                .numeric_value_at(row + col * rows)
                .ok_or_else(|| pwelch_error(&PWELCH_ERROR_INTERNAL))?;
            column.push(Complex::new(scalar_f64(re), scalar_f64(im)));
            if let Some(units) = work.record(1) {
                cooperation.completed(units).await?;
            }
        }
        columns.push(column);
    }
    work.finish(cooperation).await?;
    Ok(SignalColumns {
        columns,
        rows,
        cols,
        is_complex,
        output_precision,
    })
}

fn scalar_f64(value: runmat_value::NumericScalar) -> f64 {
    match value {
        runmat_value::NumericScalar::F64(value) => value,
        runmat_value::NumericScalar::F32(value) => f64::from(value),
        value => value
            .into_int_value()
            .expect("non-floating numeric scalar is integer")
            .to_f64(),
    }
}
