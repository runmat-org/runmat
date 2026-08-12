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
pub enum CumulativeExtrema {
    Min,
    Max,
}

pub fn cumulative_extrema(
    storage: NumericStorage,
    shape: Vec<usize>,
    dim: usize,
    direction: CumulativeDirection,
    nan_mode: CumulativeNanMode,
    extrema: CumulativeExtrema,
) -> Result<(Tensor, Tensor), String> {
    if dim == 0 {
        return Err("dimension must be >= 1".to_string());
    }
    if storage.is_empty() || dim > shape.len() {
        let indices = if storage.is_empty() {
            Tensor::new(Vec::new(), shape.clone())?
        } else {
            Tensor::new(vec![1.0; storage.len()], shape.clone())?
        };
        return Ok((Tensor::from_numeric_storage(storage, shape)?, indices));
    }

    let dim_index = dim - 1;
    let segment_len = shape[dim_index];
    if segment_len == 0 {
        let indices = Tensor::new(Vec::new(), shape.clone())?;
        return Ok((Tensor::from_numeric_storage(storage, shape)?, indices));
    }

    match storage {
        NumericStorage::F64(values) => cumulative_extrema_typed(
            values,
            shape,
            dim,
            direction,
            nan_mode,
            extrema,
            f64::NAN,
            f64::is_nan,
            NumericStorage::F64,
        ),
        NumericStorage::F32(values) => cumulative_extrema_typed(
            values,
            shape,
            dim,
            direction,
            nan_mode,
            extrema,
            f32::NAN,
            f32::is_nan,
            NumericStorage::F32,
        ),
        _ => Err("floating cumulative extrema received integer storage".to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
fn cumulative_extrema_typed<T>(
    values: Vec<T>,
    shape: Vec<usize>,
    dim: usize,
    direction: CumulativeDirection,
    nan_mode: CumulativeNanMode,
    extrema: CumulativeExtrema,
    nan: T,
    is_nan: fn(T) -> bool,
    wrap: fn(Vec<T>) -> NumericStorage,
) -> Result<(Tensor, Tensor), String>
where
    T: Copy + PartialOrd,
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
            let mut current: Option<(T, usize)> = None;
            let mut fixed_nan_index = None;
            match direction {
                CumulativeDirection::Forward => {
                    for offset in 0..segment_len {
                        let index = base + before + offset * stride_before;
                        let position = offset + 1;
                        let (value, value_index) = update_extrema(
                            values[index],
                            position,
                            &mut current,
                            &mut fixed_nan_index,
                            nan_mode,
                            extrema,
                            nan,
                            is_nan,
                        );
                        values_out[index] = value;
                        indices_out[index] = value_index;
                    }
                }
                CumulativeDirection::Reverse => {
                    for offset in (0..segment_len).rev() {
                        let index = base + before + offset * stride_before;
                        let position = offset + 1;
                        let (value, value_index) = update_extrema(
                            values[index],
                            position,
                            &mut current,
                            &mut fixed_nan_index,
                            nan_mode,
                            extrema,
                            nan,
                            is_nan,
                        );
                        values_out[index] = value;
                        indices_out[index] = value_index;
                    }
                }
            }
        }
    }

    let values = Tensor::from_numeric_storage(wrap(values_out), shape.clone())?;
    let indices = Tensor::new(indices_out, shape)?;
    Ok((values, indices))
}

#[allow(clippy::too_many_arguments)]
fn update_extrema<T>(
    value: T,
    position: usize,
    current: &mut Option<(T, usize)>,
    fixed_nan_index: &mut Option<usize>,
    nan_mode: CumulativeNanMode,
    extrema: CumulativeExtrema,
    nan: T,
    is_nan: fn(T) -> bool,
) -> (T, f64)
where
    T: Copy + PartialOrd,
{
    if matches!(nan_mode, CumulativeNanMode::Include) {
        if let Some(index) = *fixed_nan_index {
            return (nan, index as f64);
        }
        if is_nan(value) {
            *fixed_nan_index = Some(position);
            return (nan, position as f64);
        }
    } else if is_nan(value) {
        return current
            .map(|(current, index)| (current, index as f64))
            .unwrap_or((nan, f64::NAN));
    }

    let is_better = current.is_none_or(|(current, _)| match extrema {
        CumulativeExtrema::Min => value < current,
        CumulativeExtrema::Max => value > current,
    });
    if is_better {
        *current = Some((value, position));
    }
    let (current, index) = current.expect("non-NaN value establishes cumulative extrema");
    (current, index as f64)
}
