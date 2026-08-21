use std::ops::{Add, Mul};

use runmat_value::{NumericStorage, Tensor};

#[derive(Debug, Clone, Copy)]
pub enum CumulativeDirection {
    Forward,
    Reverse,
}

#[derive(Debug, Clone, Copy)]
pub enum CumulativeNanMode {
    Include,
    Omit,
}

#[derive(Debug, Clone, Copy)]
pub enum CumulativeOperation {
    Sum,
    Product,
}

pub fn cumulative(
    storage: NumericStorage,
    shape: Vec<usize>,
    dim: usize,
    direction: CumulativeDirection,
    nan_mode: CumulativeNanMode,
    operation: CumulativeOperation,
) -> Result<Tensor, String> {
    if dim == 0 {
        return Err("dimension must be >= 1".to_string());
    }
    if storage.is_empty() || dim > shape.len() || shape[dim - 1] == 0 {
        return Tensor::from_numeric_storage(storage, shape);
    }

    match storage {
        NumericStorage::F64(values) => cumulative_typed(
            values,
            shape,
            dim,
            direction,
            nan_mode,
            operation,
            0.0f64,
            1.0f64,
            f64::NAN,
            f64::is_nan,
            NumericStorage::F64,
        ),
        NumericStorage::F32(values) => cumulative_typed(
            values,
            shape,
            dim,
            direction,
            nan_mode,
            operation,
            0.0f32,
            1.0f32,
            f32::NAN,
            f32::is_nan,
            NumericStorage::F32,
        ),
        _ => Err("floating cumulative arithmetic received integer storage".to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
fn cumulative_typed<T>(
    values: Vec<T>,
    shape: Vec<usize>,
    dim: usize,
    direction: CumulativeDirection,
    nan_mode: CumulativeNanMode,
    operation: CumulativeOperation,
    zero: T,
    one: T,
    nan: T,
    is_nan: fn(T) -> bool,
    wrap: fn(Vec<T>) -> NumericStorage,
) -> Result<Tensor, String>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    let dim_index = dim - 1;
    let segment_len = shape[dim_index];
    let stride_before = shape[..dim_index].iter().product::<usize>();
    let stride_after = shape[dim..].iter().product::<usize>();
    let block = stride_before * segment_len;
    let mut output = values.clone();

    for after in 0..stride_after {
        let base = after * block;
        for before in 0..stride_before {
            let mut accumulator = match operation {
                CumulativeOperation::Sum => zero,
                CumulativeOperation::Product => one,
            };
            let mut poisoned = false;
            match direction {
                CumulativeDirection::Forward => {
                    for offset in 0..segment_len {
                        let index = base + before + offset * stride_before;
                        output[index] = update_accumulator(
                            values[index],
                            &mut accumulator,
                            &mut poisoned,
                            nan_mode,
                            operation,
                            nan,
                            is_nan,
                        );
                    }
                }
                CumulativeDirection::Reverse => {
                    for offset in (0..segment_len).rev() {
                        let index = base + before + offset * stride_before;
                        output[index] = update_accumulator(
                            values[index],
                            &mut accumulator,
                            &mut poisoned,
                            nan_mode,
                            operation,
                            nan,
                            is_nan,
                        );
                    }
                }
            }
        }
    }

    Tensor::from_numeric_storage(wrap(output), shape)
}

fn update_accumulator<T>(
    value: T,
    accumulator: &mut T,
    poisoned: &mut bool,
    nan_mode: CumulativeNanMode,
    operation: CumulativeOperation,
    nan: T,
    is_nan: fn(T) -> bool,
) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    if matches!(nan_mode, CumulativeNanMode::Include) {
        if *poisoned || is_nan(value) {
            *poisoned = true;
            return nan;
        }
    } else if is_nan(value) {
        return *accumulator;
    }

    *accumulator = match operation {
        CumulativeOperation::Sum => *accumulator + value,
        CumulativeOperation::Product => *accumulator * value,
    };
    *accumulator
}
