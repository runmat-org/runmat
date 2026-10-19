use runmat_value::{spacing_f32, spacing_f64, NumericDType, SparseTensor, Value};

use crate::context::RuntimeContext;
use crate::BuiltinResult;

pub(super) async fn value(
    input: &SparseTensor,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    let len = input
        .rows
        .checked_mul(input.cols)
        .ok_or_else(|| super::super::errors::internal("eps", "sparse dimensions overflow"))?;
    match input.numeric_dtype() {
        Some(NumericDType::F64) => {
            if input.is_complex() {
                let values = input.as_complex_f64_slice().ok_or_else(|| {
                    super::super::errors::internal("eps", "invalid complex sparse values")
                })?;
                sparse_output_f64(input, values, len, runtime, |value| {
                    spacing_f64(value.0.hypot(value.1))
                })
                .await
            } else {
                let values = input.as_f64_slice().ok_or_else(|| {
                    super::super::errors::internal("eps", "invalid sparse values")
                })?;
                sparse_output_f64(input, values, len, runtime, |value| spacing_f64(value)).await
            }
        }
        Some(NumericDType::F32) => {
            if input.is_complex() {
                let values = input.as_complex_f32_slice().ok_or_else(|| {
                    super::super::errors::internal("eps", "invalid complex sparse values")
                })?;
                sparse_output_f32(input, values, len, runtime, |value| {
                    spacing_f32(value.0.hypot(value.1))
                })
                .await
            } else {
                let values = input.as_f32_slice().ok_or_else(|| {
                    super::super::errors::internal("eps", "invalid sparse values")
                })?;
                sparse_output_f32(input, values, len, runtime, |value| spacing_f32(value)).await
            }
        }
        _ => Err(super::super::errors::class(
            "eps",
            "eps requires a single or double input",
        )),
    }
}

async fn sparse_output_f64<T: Copy>(
    input: &SparseTensor,
    values: &[T],
    len: usize,
    runtime: Option<&RuntimeContext>,
    spacing: impl Fn(T) -> f64,
) -> BuiltinResult<Value> {
    let mut column = 0;
    let mut index = 0;
    super::output::output_f64(&[input.rows, input.cols], len, runtime, |start, output| {
        output.fill(f64::from_bits(1));
        let end = start + output.len();
        let last_column = (end - 1) / input.rows;
        let mut column_work = 0;
        let mut stored_work = 0;
        while column <= last_column {
            let column_end = input.col_ptrs[column + 1];
            while index < column_end {
                let position = input.row_indices[index] + column * input.rows;
                if position >= end {
                    break;
                }
                if position >= start {
                    output[position - start] = spacing(values[index]);
                }
                index += 1;
                stored_work += 1;
            }
            if index < column_end {
                break;
            }
            column += 1;
            column_work += 1;
        }
        output.len() + column_work + stored_work
    })
    .await
}

async fn sparse_output_f32<T: Copy>(
    input: &SparseTensor,
    values: &[T],
    len: usize,
    runtime: Option<&RuntimeContext>,
    spacing: impl Fn(T) -> f32,
) -> BuiltinResult<Value> {
    let mut column = 0;
    let mut index = 0;
    super::output::output_f32(&[input.rows, input.cols], len, runtime, |start, output| {
        output.fill(f32::from_bits(1));
        let end = start + output.len();
        let last_column = (end - 1) / input.rows;
        let mut column_work = 0;
        let mut stored_work = 0;
        while column <= last_column {
            let column_end = input.col_ptrs[column + 1];
            while index < column_end {
                let position = input.row_indices[index] + column * input.rows;
                if position >= end {
                    break;
                }
                if position >= start {
                    output[position - start] = spacing(values[index]);
                }
                index += 1;
                stored_work += 1;
            }
            if index < column_end {
                break;
            }
            column += 1;
            column_work += 1;
        }
        output.len() + column_work + stored_work
    })
    .await
}
