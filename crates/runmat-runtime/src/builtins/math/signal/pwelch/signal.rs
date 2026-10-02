use super::*;

pub(super) async fn value_to_signal_columns(value: Value) -> BuiltinResult<SignalColumns> {
    match value {
        Value::GpuTensor(handle) => {
            let provider = super::super::resident_output::owner(&handle, || {
                pwelch_error(&PWELCH_ERROR_INTERNAL)
            })?;
            let value =
                crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
                    provider, &handle,
                )
                .await?;
            match value {
                Value::Tensor(tensor) => tensor_to_signal_columns(tensor),
                Value::ComplexTensor(tensor) => complex_tensor_to_signal_columns(tensor),
                Value::LogicalArray(logical) => {
                    let tensor = crate::builtins::common::tensor::logical_to_tensor(&logical)
                        .map_err(|err| {
                            pwelch_error_with_detail(&PWELCH_ERROR_INVALID_SIGNAL, err)
                        })?;
                    tensor_to_signal_columns(tensor)
                }
                Value::Num(value) => tensor_to_signal_columns(
                    Tensor::new(vec![value], handle.shape().to_vec())
                        .map_err(|cause| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, cause))?,
                ),
                other => Err(pwelch_error_with_detail(
                    &PWELCH_ERROR_INVALID_SIGNAL,
                    format!("unexpected downloaded signal {other:?}"),
                )),
            }
        }
        Value::Tensor(tensor) => tensor_to_signal_columns(tensor),
        Value::ComplexTensor(tensor) => complex_tensor_to_signal_columns(tensor),
        Value::LogicalArray(logical) => {
            let tensor = crate::builtins::common::tensor::logical_to_tensor(&logical)
                .map_err(|err| pwelch_error_with_detail(&PWELCH_ERROR_INVALID_SIGNAL, err))?;
            tensor_to_signal_columns(tensor)
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

pub(super) fn tensor_to_signal_columns(tensor: Tensor) -> BuiltinResult<SignalColumns> {
    if tensor.shape.len() > 2 {
        return Err(pwelch_error_with_detail(
            &PWELCH_ERROR_INVALID_SIGNAL,
            "x must be a vector or 2-D matrix",
        ));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    let output_precision = SignalPrecision::from_numeric_dtype(tensor.numeric_dtype());
    if rows == 1 || cols == 1 {
        let values = tensor_utils::tensor_into_values_f64(tensor);
        return Ok(SignalColumns {
            columns: vec![values
                .into_iter()
                .map(|value| Complex::new(value, 0.0))
                .collect()],
            rows: rows.max(cols),
            cols: 1,
            is_complex: false,
            output_precision,
        });
    }
    let mut columns = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut column = Vec::with_capacity(rows);
        for row in 0..rows {
            column.push(Complex::new(
                tensor_utils::tensor_value_f64(&tensor, row + col * rows),
                0.0,
            ));
        }
        columns.push(column);
    }
    Ok(SignalColumns {
        columns,
        rows,
        cols,
        is_complex: false,
        output_precision,
    })
}

pub(super) fn complex_tensor_to_signal_columns(
    tensor: runmat_value::ComplexTensor,
) -> BuiltinResult<SignalColumns> {
    if tensor.shape.len() > 2 {
        return Err(pwelch_error_with_detail(
            &PWELCH_ERROR_INVALID_SIGNAL,
            "x must be a vector or 2-D matrix",
        ));
    }
    let rows = tensor.rows;
    let cols = tensor.cols;
    let output_precision = SignalPrecision::from_numeric_dtype(tensor.numeric_dtype());
    let is_complex = true;
    let values = tensor.materialize_f64();
    if rows == 1 || cols == 1 {
        return Ok(SignalColumns {
            columns: vec![values
                .into_iter()
                .map(|(re, im)| Complex::new(re, im))
                .collect()],
            rows: rows.max(cols),
            cols: 1,
            is_complex,
            output_precision,
        });
    }
    let mut columns = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut column = Vec::with_capacity(rows);
        for row in 0..rows {
            let (re, im) = values[row + col * rows];
            column.push(Complex::new(re, im));
        }
        columns.push(column);
    }
    Ok(SignalColumns {
        columns,
        rows,
        cols,
        is_complex,
        output_precision,
    })
}
