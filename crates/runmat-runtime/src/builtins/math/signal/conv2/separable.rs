//! Two-pass separable two-dimensional convolution.

use num_complex::Complex;

use super::{
    conv2_allocation_error, conv2_error, conv2_matrices, outer_product, Conv2Mode, Matrix,
    CONV2_ERROR_BUILD_OUTPUT,
};
use crate::builtins::math::signal::common::{SignalCooperation, SIGNAL_WORK_CHUNK};
use crate::BuiltinResult;

mod balance;
use balance::balanced_factors;

#[derive(Clone, Copy)]
enum Axis {
    Rows,
    Columns,
}

pub(super) async fn convolve_separable(
    signal: &Matrix,
    column: &[Complex<f64>],
    row: &[Complex<f64>],
    mode: Conv2Mode,
    single_output: bool,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Matrix> {
    if signal.is_empty() || column.is_empty() || row.is_empty() {
        return match mode {
            Conv2Mode::Same => Matrix::zeros(signal.rows, signal.cols, cooperation).await,
            Conv2Mode::Full | Conv2Mode::Valid => Matrix::zeros(0, 0, cooperation).await,
        };
    }
    if mode == Conv2Mode::Valid && (signal.rows < column.len() || signal.cols < row.len()) {
        return Matrix::zeros(0, 0, cooperation).await;
    }
    let finite_result = {
        let Some(factors) =
            balanced_factors(signal, column, row, single_output, cooperation).await?
        else {
            // An unrepresentable balance is not an excuse to erase a nonzero
            // coefficient. The general kernel retains the prior numerical route.
            return general_kernel(signal, column, row, mode, single_output, cooperation).await;
        };
        let vertical = convolve_axis(
            signal,
            factors.column.as_ref(),
            Axis::Rows,
            single_output,
            cooperation,
        )
        .await?;
        if !all_finite(&vertical, cooperation).await? {
            None
        } else {
            let full = convolve_axis(
                &vertical,
                factors.row.as_ref(),
                Axis::Columns,
                single_output,
                cooperation,
            )
            .await?;
            all_finite(&full, cooperation).await?.then_some(full)
        }
    };
    let Some(full) = finite_result else {
        return general_kernel(signal, column, row, mode, single_output, cooperation).await;
    };
    match mode {
        Conv2Mode::Full => Ok(full),
        Conv2Mode::Same => {
            let row_start = column.len() / 2;
            let col_start = row.len() / 2;
            full.slice(
                row_start,
                row_start + signal.rows,
                col_start,
                col_start + signal.cols,
                cooperation,
            )
            .await
        }
        Conv2Mode::Valid => {
            let rows = signal.rows - column.len() + 1;
            let cols = signal.cols - row.len() + 1;
            let row_start = column.len() - 1;
            let col_start = row.len() - 1;
            full.slice(
                row_start,
                row_start + rows,
                col_start,
                col_start + cols,
                cooperation,
            )
            .await
        }
    }
}

async fn general_kernel(
    signal: &Matrix,
    column: &[Complex<f64>],
    row: &[Complex<f64>],
    mode: Conv2Mode,
    single_output: bool,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Matrix> {
    let kernel = outer_product(column, row, single_output, cooperation).await?;
    conv2_matrices(signal, &kernel, mode, single_output, cooperation).await
}

async fn all_finite(matrix: &Matrix, cooperation: &mut SignalCooperation) -> BuiltinResult<bool> {
    for chunk in matrix.data.chunks(SIGNAL_WORK_CHUNK as usize) {
        if chunk
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Ok(false);
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(true)
}

async fn convolve_axis(
    signal: &Matrix,
    kernel: &[Complex<f64>],
    axis: Axis,
    single_output: bool,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Matrix> {
    let rows = if matches!(axis, Axis::Rows) {
        signal
            .rows
            .checked_add(kernel.len())
            .and_then(|n| n.checked_sub(1))
    } else {
        Some(signal.rows)
    }
    .ok_or_else(|| conv2_error(&CONV2_ERROR_BUILD_OUTPUT))?;
    let cols = if matches!(axis, Axis::Columns) {
        signal
            .cols
            .checked_add(kernel.len())
            .and_then(|n| n.checked_sub(1))
    } else {
        Some(signal.cols)
    }
    .ok_or_else(|| conv2_error(&CONV2_ERROR_BUILD_OUTPUT))?;
    let mut output = Matrix::zeros(rows, cols, cooperation).await?;
    let mut work = 0_u64;
    for col in 0..signal.cols {
        for row in 0..signal.rows {
            let sample = signal.get(row, col);
            for (offset, coefficient) in kernel.iter().copied().enumerate() {
                let (out_row, out_col) = match axis {
                    Axis::Rows => (row + offset, col),
                    Axis::Columns => (row, col + offset),
                };
                if single_output {
                    let sample = Complex::<f32>::new(sample.re as f32, sample.im as f32);
                    let coefficient =
                        Complex::<f32>::new(coefficient.re as f32, coefficient.im as f32);
                    let index = output.index(out_row, out_col);
                    let current = output.data[index];
                    let current = Complex::<f32>::new(current.re as f32, current.im as f32);
                    let value = current + sample * coefficient;
                    output.data[index] = Complex::new(f64::from(value.re), f64::from(value.im));
                } else {
                    output.add_assign(out_row, out_col, sample * coefficient);
                }
                work += 1;
                if work == SIGNAL_WORK_CHUNK {
                    cooperation.completed(work).await?;
                    work = 0;
                }
            }
        }
    }
    cooperation.completed(work).await?;
    Ok(output)
}

#[cfg(test)]
mod tests;
