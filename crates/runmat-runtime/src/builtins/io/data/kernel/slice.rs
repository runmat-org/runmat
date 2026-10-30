//! Dataset slice domain helpers.

use super::*;

pub(in crate::builtins::io::data) struct DimRange {
    pub(in crate::builtins::io::data) start: usize,
    pub(in crate::builtins::io::data) end: usize,
}

pub(in crate::builtins::io::data) async fn read_slice_payload(
    payload: &DataArrayPayload,
    slice_spec: &Value,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<DataArrayPayload> {
    crate::data::cooperate_data_work(budget, 0).await?;
    let ranges = parse_slice_spec(slice_spec, &payload.shape)?;
    let out_shape: Vec<usize> = ranges
        .iter()
        .map(|r| r.end.saturating_sub(r.start))
        .collect();
    let mut out_values = crate::data::DataArrayValues::zeros(&payload.dtype, 0)?;
    let mut out_imaginary_values = payload
        .imaginary_values
        .as_ref()
        .map(|_| crate::data::DataArrayValues::zeros(&payload.dtype, 0))
        .transpose()?;
    let mut out_index = vec![0usize; out_shape.len()];
    loop {
        crate::data::cooperate_data_work(budget, 1).await?;
        let source_index: Vec<usize> = out_index
            .iter()
            .enumerate()
            .map(|(dim, idx)| ranges[dim].start + *idx)
            .collect();
        let linear = linear_index_column_major(&source_index, &payload.shape)?;
        out_values.push(payload.values.get(linear)?)?;
        if let (Some(source), Some(target)) = (&payload.imaginary_values, &mut out_imaginary_values)
        {
            target.push(source.get(linear)?)?;
        }

        if !advance_index(&mut out_index, &out_shape) {
            break;
        }
    }
    Ok(DataArrayPayload {
        dtype: payload.dtype.clone(),
        shape: out_shape,
        values: out_values,
        imaginary_values: out_imaginary_values,
    })
}

pub(in crate::builtins::io::data) async fn write_slice_payload(
    payload: &DataArrayPayload,
    slice_spec: &Value,
    rhs: &Value,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<DataArrayPayload> {
    crate::data::cooperate_data_work(budget, 0).await?;
    let ranges = parse_slice_spec(slice_spec, &payload.shape)?;
    let target_shape: Vec<usize> = ranges
        .iter()
        .map(|r| r.end.saturating_sub(r.start))
        .collect();
    let rhs = DataArrayPayload::from_value(payload.dtype.clone(), rhs)?;
    if rhs.shape != target_shape {
        return Err(data_error(format!(
            "SHAPE_MISMATCH: rhs shape {:?} must match target slice shape {:?}",
            rhs.shape, target_shape
        )));
    }

    let mut next = payload.values.clone();
    if payload.imaginary_values.is_some() != rhs.imaginary_values.is_some() {
        return Err(data_error(
            "CLASS_MISMATCH: real and complex data-array slices cannot be assigned to one another",
        ));
    }
    let mut next_imaginary = payload.imaginary_values.clone();
    let mut rhs_index = vec![0usize; target_shape.len()];
    let mut rhs_linear = 0usize;
    loop {
        crate::data::cooperate_data_work(budget, 1).await?;
        let target_index: Vec<usize> = rhs_index
            .iter()
            .enumerate()
            .map(|(dim, idx)| ranges[dim].start + *idx)
            .collect();
        let target_linear = linear_index_column_major(&target_index, &payload.shape)?;
        next.set(target_linear, rhs.values.get(rhs_linear)?)?;
        if let (Some(target), Some(source)) = (&mut next_imaginary, &rhs.imaginary_values) {
            target.set(target_linear, source.get(rhs_linear)?)?;
        }
        rhs_linear += 1;

        if !advance_index(&mut rhs_index, &target_shape) {
            break;
        }
    }

    Ok(DataArrayPayload {
        dtype: payload.dtype.clone(),
        shape: payload.shape.clone(),
        values: next,
        imaginary_values: next_imaginary,
    })
}

pub(in crate::builtins::io::data) fn parse_slice_spec(
    slice_spec: &Value,
    shape: &[usize],
) -> BuiltinResult<Vec<DimRange>> {
    match slice_spec {
        Value::Cell(cell) => {
            if cell.data().is_empty() {
                return Err(data_error("INVALID_SLICE: empty slice specification"));
            }
            let mut ranges = Vec::with_capacity(shape.len());
            for (dim, extent) in shape.iter().enumerate() {
                if let Some(item) = cell.data().get(dim) {
                    ranges.push(parse_dim_range(item, *extent)?);
                } else {
                    ranges.push(DimRange {
                        start: 0,
                        end: *extent,
                    });
                }
            }
            Ok(ranges)
        }
        Value::String(runmat_value::StringElement::Text(s)) if s == ":" => Ok(shape
            .iter()
            .map(|extent| DimRange {
                start: 0,
                end: *extent,
            })
            .collect()),
        _ => Err(data_error(
            "INVALID_SLICE: slice must be a cell spec like {1:10, :} or ':'",
        )),
    }
}

pub(in crate::builtins::io::data) fn parse_dim_range(
    value: &Value,
    extent: usize,
) -> BuiltinResult<DimRange> {
    if extent == 0 {
        return Ok(DimRange { start: 0, end: 0 });
    }
    match value {
        Value::String(runmat_value::StringElement::Text(s)) if s == ":" => Ok(DimRange {
            start: 0,
            end: extent,
        }),
        Value::Num(n) => {
            let idx = floating_dimension_to_usize(*n)
                .and_then(|index| index.checked_sub(1))
                .ok_or_else(|| data_error("INVALID_SLICE: index out of bounds"))?;
            if idx >= extent {
                return Err(data_error("INVALID_SLICE: index out of bounds"));
            }
            Ok(DimRange {
                start: idx,
                end: idx + 1,
            })
        }
        Value::Int(i) => {
            let Some(idx) = i.try_to_usize().and_then(|index| index.checked_sub(1)) else {
                return Err(data_error("INVALID_SLICE: index out of bounds"));
            };
            if idx >= extent {
                return Err(data_error("INVALID_SLICE: index out of bounds"));
            }
            Ok(DimRange {
                start: idx,
                end: idx + 1,
            })
        }
        Value::Tensor(t) if tensor_utils::tensor_element_len(t) == 2 => {
            let start = t
                .numeric_value_at(0)
                .and_then(numeric_dimension_to_usize)
                .and_then(|index| index.checked_sub(1))
                .ok_or_else(|| data_error("INVALID_SLICE: range out of bounds"))?;
            let end_inclusive = t
                .numeric_value_at(1)
                .and_then(numeric_dimension_to_usize)
                .and_then(|index| index.checked_sub(1))
                .ok_or_else(|| data_error("INVALID_SLICE: range out of bounds"))?;
            if end_inclusive < start || end_inclusive >= extent {
                return Err(data_error("INVALID_SLICE: range out of bounds"));
            }
            Ok(DimRange {
                start,
                end: end_inclusive + 1,
            })
        }
        _ => Err(data_error(
            "INVALID_SLICE: dimension must be ':', scalar index, or [start end] range",
        )),
    }
}

pub(in crate::builtins::io::data) fn linear_index_column_major(
    index: &[usize],
    shape: &[usize],
) -> BuiltinResult<usize> {
    if index.len() != shape.len() {
        return Err(data_error("INVALID_SLICE: rank mismatch"));
    }
    let mut stride = 1usize;
    let mut linear = 0usize;
    for (idx, extent) in index.iter().zip(shape.iter()) {
        if *idx >= *extent {
            return Err(data_error("INVALID_SLICE: index out of bounds"));
        }
        linear += idx * stride;
        stride = stride.saturating_mul(*extent);
    }
    Ok(linear)
}

pub(in crate::builtins::io::data) fn advance_index(index: &mut [usize], shape: &[usize]) -> bool {
    if shape.is_empty() {
        return false;
    }
    for dim in 0..shape.len() {
        index[dim] += 1;
        if index[dim] < shape[dim] {
            return true;
        }
        index[dim] = 0;
    }
    false
}

pub(in crate::builtins::io::data) fn chunk_key(coords: &[usize]) -> String {
    coords
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(".")
}

pub(in crate::builtins::io::data) fn chunk_start_for_coords(
    coords: &[usize],
    chunk_shape: &[usize],
) -> Vec<usize> {
    coords
        .iter()
        .enumerate()
        .map(|(dim, coord)| coord * chunk_shape.get(dim).copied().unwrap_or(1).max(1))
        .collect()
}

pub(in crate::builtins::io::data) fn chunk_extent_for_start(
    start: &[usize],
    chunk_shape: &[usize],
    shape: &[usize],
) -> Vec<usize> {
    start
        .iter()
        .enumerate()
        .map(|(dim, start)| {
            let chunk = chunk_shape.get(dim).copied().unwrap_or(1).max(1);
            let end = (*start + chunk).min(shape[dim]);
            end.saturating_sub(*start)
        })
        .collect()
}

pub(in crate::builtins::io::data) fn chunk_intersection(
    ranges: &[DimRange],
    chunk_start: &[usize],
    chunk_extent: &[usize],
) -> Vec<DimRange> {
    let mut out = Vec::with_capacity(ranges.len());
    for dim in 0..ranges.len() {
        let c_start = chunk_start[dim];
        let c_end = c_start + chunk_extent[dim];
        let start = ranges[dim].start.max(c_start);
        let end = ranges[dim].end.min(c_end);
        if start >= end {
            return Vec::new();
        }
        out.push(DimRange { start, end });
    }
    out
}

pub(in crate::builtins::io::data) fn touched_chunk_coords(
    ranges: &[DimRange],
    chunk_shape: &[usize],
    shape: &[usize],
) -> Vec<Vec<usize>> {
    let mut span = Vec::with_capacity(ranges.len());
    let mut begin = Vec::with_capacity(ranges.len());
    for dim in 0..ranges.len() {
        if shape[dim] == 0 {
            return Vec::new();
        }
        let chunk = chunk_shape.get(dim).copied().unwrap_or(1).max(1);
        let first = ranges[dim].start / chunk;
        let last = (ranges[dim].end.saturating_sub(1)) / chunk;
        begin.push(first);
        span.push(last.saturating_sub(first) + 1);
    }
    let mut local = vec![0usize; span.len()];
    let mut out = Vec::new();
    loop {
        out.push(
            local
                .iter()
                .enumerate()
                .map(|(dim, v)| begin[dim] + *v)
                .collect::<Vec<_>>(),
        );
        if !advance_index(&mut local, &span) {
            break;
        }
    }
    out
}
