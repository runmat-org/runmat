use num_traits::Float;
use runmat_value::{ComplexStorage, ComplexTensor, Tensor};

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Forward,
    Reverse,
}

#[derive(Debug, Clone, Copy)]
pub enum NanMode {
    Include,
    Omit,
}

#[derive(Debug, Clone, Copy)]
pub enum Extrema {
    Min,
    Max,
}

pub fn cumulative_extrema(
    storage: ComplexStorage,
    shape: Vec<usize>,
    dim: usize,
    direction: Direction,
    nan_mode: NanMode,
    extrema: Extrema,
) -> Result<(ComplexTensor, Tensor), String> {
    if dim == 0 {
        return Err("dimension must be >= 1".to_string());
    }
    if matches!(storage, ComplexStorage::Integer(_)) {
        return Err("complex cumulative extrema received integer storage".to_string());
    }
    if storage.is_empty() || dim > shape.len() {
        let indices = if storage.is_empty() {
            Tensor::new(Vec::new(), shape.clone())?
        } else {
            Tensor::new(vec![1.0; storage.len()], shape.clone())?
        };
        return Ok((
            ComplexTensor::from_complex_storage(storage, shape)?,
            indices,
        ));
    }
    if shape[dim - 1] == 0 {
        let indices = Tensor::new(Vec::new(), shape.clone())?;
        return Ok((
            ComplexTensor::from_complex_storage(storage, shape)?,
            indices,
        ));
    }

    match storage {
        ComplexStorage::F64(values) => cumulative_extrema_typed(
            values,
            shape,
            dim,
            direction,
            nan_mode,
            extrema,
            ComplexStorage::F64,
        ),
        ComplexStorage::F32(values) => cumulative_extrema_typed(
            values,
            shape,
            dim,
            direction,
            nan_mode,
            extrema,
            ComplexStorage::F32,
        ),
        ComplexStorage::Integer(_) => unreachable!("integer storage rejected above"),
    }
}

fn cumulative_extrema_typed<T>(
    values: Vec<(T, T)>,
    shape: Vec<usize>,
    dim: usize,
    direction: Direction,
    nan_mode: NanMode,
    extrema: Extrema,
    wrap: fn(Vec<(T, T)>) -> ComplexStorage,
) -> Result<(ComplexTensor, Tensor), String>
where
    T: Float,
{
    let dim_index = dim - 1;
    let segment_len = shape[dim_index];
    let stride_before = shape[..dim_index].iter().product::<usize>();
    let stride_after = shape[dim..].iter().product::<usize>();
    let block = stride_before * segment_len;
    let mut values_out = values.clone();
    let mut indices_out = vec![0.0; values.len()];

    for after in 0..stride_after {
        let base = after * block;
        for before in 0..stride_before {
            match direction {
                Direction::Forward => scan_segment(
                    0..segment_len,
                    base,
                    before,
                    stride_before,
                    &values,
                    &mut values_out,
                    &mut indices_out,
                    nan_mode,
                    extrema,
                ),
                Direction::Reverse => scan_segment(
                    (0..segment_len).rev(),
                    base,
                    before,
                    stride_before,
                    &values,
                    &mut values_out,
                    &mut indices_out,
                    nan_mode,
                    extrema,
                ),
            }
        }
    }

    Ok((
        ComplexTensor::from_complex_storage(wrap(values_out), shape.clone())?,
        Tensor::new(indices_out, shape)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn scan_segment<T>(
    offsets: impl Iterator<Item = usize>,
    base: usize,
    before: usize,
    stride_before: usize,
    values: &[(T, T)],
    values_out: &mut [(T, T)],
    indices_out: &mut [f64],
    nan_mode: NanMode,
    extrema: Extrema,
) where
    T: Float,
{
    let mut current: Option<((T, T), usize)> = None;
    let mut fixed_nan_index = None;
    for offset in offsets {
        let index = base + before + offset * stride_before;
        let position = offset + 1;
        let value = values[index];
        if matches!(nan_mode, NanMode::Include) {
            if let Some(nan_index) = fixed_nan_index {
                values_out[index] = complex_nan();
                indices_out[index] = nan_index as f64;
                continue;
            }
            if complex_is_nan(value) {
                fixed_nan_index = Some(position);
                values_out[index] = complex_nan();
                indices_out[index] = position as f64;
                continue;
            }
        } else if complex_is_nan(value) {
            if let Some((current_value, current_index)) = current {
                values_out[index] = current_value;
                indices_out[index] = current_index as f64;
            } else {
                values_out[index] = complex_nan();
                indices_out[index] = f64::NAN;
            }
            continue;
        }

        let better = current.is_none_or(|(current, _)| match extrema {
            Extrema::Min => complex_less(value, current),
            Extrema::Max => complex_less(current, value),
        });
        if better {
            current = Some((value, position));
        }
        let (current_value, current_index) =
            current.expect("non-NaN value establishes cumulative extrema");
        values_out[index] = current_value;
        indices_out[index] = current_index as f64;
    }
}

fn complex_less<T: Float>(left: (T, T), right: (T, T)) -> bool {
    let left_magnitude = left.0 * left.0 + left.1 * left.1;
    let right_magnitude = right.0 * right.0 + right.1 * right.1;
    left_magnitude < right_magnitude
        || (left_magnitude == right_magnitude && left.1.atan2(left.0) < right.1.atan2(right.0))
}

fn complex_is_nan<T: Float>(value: (T, T)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn complex_nan<T: Float>() -> (T, T) {
    (T::nan(), T::nan())
}
